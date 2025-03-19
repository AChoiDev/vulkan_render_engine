use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderingAttachmentInfo, RenderingInfo,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features,
        QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, Version, VulkanError, VulkanLibrary,
};

use std::sync::{mpsc::Receiver, Arc};

pub fn run(
    surface_instance_exts : InstanceExtensions,
    shutdown_receiver: Receiver<()>,
    window_receiver: Receiver<winit::window::Window>) 
    -> Result<(), Box<dyn std::error::Error>> {

    let window = Arc::new(window_receiver.recv().expect("Failed to receive initial window"));

    let vk_lib = VulkanLibrary::new()?;
    let req_instance_extensions = surface_instance_exts;

    let vk_instance = Instance::new(
        vk_lib,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: req_instance_extensions,
            ..Default::default()
        }
    )?;


    let swapchain_ext = DeviceExtensions { 
        khr_swapchain: true, 
        ..DeviceExtensions::empty()
    };

    let surface = Surface::from_window(vk_instance.clone(), window.clone()).unwrap();

    let (physical_device, queue_family_index) = vk_instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
        })
        .filter(|p| {
            p.supported_extensions().contains(&swapchain_ext)
        })
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .ok_or("Could not find a suitable physical device")?;

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    ); 

    let device_extensions = {
        if physical_device.api_version() < Version::V1_3 {
            DeviceExtensions {
                khr_dynamic_rendering: true,
                ..swapchain_ext
            }
        } else {
            swapchain_ext
        }
    };

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            enabled_features: Features {
                dynamic_rendering: true,
                ..Features::empty()
            },
            ..Default::default()
        }
    )?;

    let queue = queues.next()
        .ok_or("Could not find a suitable queue from physical device")?;

    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                // Some drivers report an `min_image_count` of 1, but fullscreen mode requires at
                // least 2. Therefore we must ensure the count is at least 2, otherwise the program
                // would crash when entering fullscreen mode on those drivers.
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,

                // The size of the window, only used to initially setup the swapchain.
                //
                // NOTE:
                // On some drivers the swapchain extent is specified by
                // `surface_capabilities.current_extent` and the swapchain size must use this
                // extent. This extent is always the same as the window size.
                //
                // However, other drivers don't specify a value, i.e.
                // `surface_capabilities.current_extent` is `None`. These drivers will allow
                // anything, but the only sensible value is the window size.
                //
                // Both of these cases need the swapchain to use the window size, so we just
                // use that.
                image_extent: window.inner_size().into(),

                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )?
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // We now create a buffer that will store the shape of our triangle. We use `#[repr(C)]` here
    // to force rustc to use a defined layout for our data, as the default representation has *no
    // guarantees*.
    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

    let vertices = [
        Vertex {
            position: [-0.5, -0.25],
        },
        Vertex {
            position: [0.0, 0.5],
        },
        Vertex {
            position: [0.25, -0.1],
        },
    ];
    let vertex_buffer = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices,
    )
    .unwrap();

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450

                layout(location = 0) in vec2 position;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            ",
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            ",
        }
    }

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // Before we draw, we have to create what is called a **pipeline**. A pipeline describes how
    // a GPU operation is to be performed. It is similar to an OpenGL program, but it also contains
    // many settings for customization, all baked into a single object. For drawing, we create
    // a **graphics** pipeline, but there are also other types of pipeline.
    let pipeline = {
        // First, we load the shaders that the pipeline will use:
        // the vertex shader and the fragment shader.
        //
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify which
        // one.
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        // Automatically generate a vertex input state from the vertex shader's input interface,
        // that takes a single vertex buffer containing `Vertex` structs.
        let vertex_input_state = Vertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();

        // Make a list of the shader stages that the pipeline will have.
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        // We must now create a **pipeline layout** object, which describes the locations and types of
        // descriptor sets and push constants used by the shaders in the pipeline.
        //
        // Multiple pipelines can share a common layout object, which is more efficient.
        // The shaders in a pipeline must use a subset of the resources described in its pipeline
        // layout, but the pipeline layout is allowed to contain resources that are not present in the
        // shaders; they can be used by shaders in other pipelines that share the same layout.
        // Thus, it is a good idea to design shaders so that many pipelines have common resource
        // locations, which allows them to share pipeline layouts.
        let layout = PipelineLayout::new(
            device.clone(),
            // Since we only have one pipeline in this example, and thus one pipeline layout,
            // we automatically generate the creation info for it from the resources used in the
            // shaders. In a real application, you would specify this information manually so that you
            // can re-use one layout in multiple pipelines.
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        // We describe the formats of attachment images where the colors, depth and/or stencil
        // information will be written. The pipeline will only be usable with this particular
        // configuration of the attachment images.
        let subpass = PipelineRenderingCreateInfo {
            // We specify a single color attachment that will be rendered to. When we begin
            // rendering, we will specify a swapchain image to be used as this attachment, so here
            // we set its format to be the same format as the swapchain.
            color_attachment_formats: vec![Some(swapchain.image_format())],
            ..Default::default()
        };

        // Finally, create the pipeline.
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                // How vertex data is read from the vertex buffers into the vertex shader.
                vertex_input_state: Some(vertex_input_state),
                // How vertices are arranged into primitive shapes.
                // The default primitive shape is a triangle.
                input_assembly_state: Some(InputAssemblyState::default()),
                // How primitives are transformed and clipped to fit the framebuffer.
                // We use a resizable viewport, set to draw over the entire window.
                viewport_state: Some(ViewportState::default()),
                // How polygons are culled and converted into a raster of pixels.
                // The default value does not perform any culling.
                rasterization_state: Some(RasterizationState::default()),
                // How multiple fragment shader samples are converted to a single pixel value.
                // The default value does not perform any multisampling.
                multisample_state: Some(MultisampleState::default()),
                // How pixel values are combined with the values already present in the framebuffer.
                // The default value overwrites the old value with the new one, without any blending.
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.color_attachment_formats.len() as u32,
                    ColorBlendAttachmentState::default(),
                )),
                // Dynamic states allows us to specify parts of the pipeline settings when
                // recording the command buffer, before we perform drawing.
                // Here, we specify that the viewport should be dynamic.
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0,
    };

    let mut attachment_image_views = window_size_dependent_setup(&images, &mut viewport);

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    println!("Renderer Started");

    loop {
        // TODO: replace shutdown receiver with only 1 shutdown code path

        // Break look if shutdown initiated
        match shutdown_receiver.try_recv() {
            Ok(_) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                println!("Shutdown initiated");
                break;
            },
            _ => {}
        }

        let image_extent: [u32; 2] = window.inner_size().into();

        if image_extent.contains(&0) {
            continue;
        }

        previous_frame_end.as_mut().unwrap().cleanup_finished();

        if recreate_swapchain {
            let (new_swapchain, new_images) = swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent,
                    ..swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            swapchain = new_swapchain;

            attachment_image_views =
                window_size_dependent_setup(&new_images, &mut viewport);

            recreate_swapchain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    recreate_swapchain = true;
                    println!("swapchain was out of date");
                    continue;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            // Before we can draw, we have to *enter a render pass*. We specify which
            // attachments we are going to use for rendering here, which needs to match
            // what was previously specified when creating the pipeline.
            .begin_rendering(RenderingInfo {
                // As before, we specify one color attachment, but now we specify the image
                // view to use as well as how it should be used.
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    // `Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of rendering.
                    load_op: AttachmentLoadOp::Clear,
                    // `Store` means that we ask the GPU to store the rendered output in
                    // the attachment image. We could also ask it to discard the result.
                    store_op: AttachmentStoreOp::Store,
                    // The value to clear the attachment with. Here we clear it with a blue
                    // color.
                    //
                    // Only attachments that have `AttachmentLoadOp::Clear` are provided
                    // with clear values, any others should use `None` as the clear value.
                    clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(
                        // We specify image view corresponding to the currently acquired
                        // swapchain image, to use for this attachment.
                        attachment_image_views[image_index as usize].clone(),
                    )
                })],
                ..Default::default()
            })
            .unwrap()
            // We are now inside the first subpass of the render pass.
            //
            // TODO: Document state setting and how it affects subsequent draw commands.
            .set_viewport(0, [viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap()
            // We add a draw command.
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            // We leave the render pass.
            .end_rendering()
            .unwrap();

        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build().unwrap();

        let future = previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
        }
    }
    
    return Ok(());
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    viewport: &mut Viewport,
) -> Vec<Arc<ImageView>> {
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>()
}