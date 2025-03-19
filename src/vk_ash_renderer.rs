use std::{collections::{HashMap, HashSet}, sync::{mpsc::Receiver, Arc}};

use ash::vk;
use vk_material_manager::MaterialManager;
use vk_rgba_manager::PowerTwoTextureLength;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use mortal::Mortal;

mod vk_rgba_manager;
mod vk_mesh_manager;
mod vk_slab_manager;

mod vk_bump_manager;
mod vk_material_manager;

use vk_mesh_manager::{BoneVertex, GeometryAllocationView, IndexedVerticesAllocation, RenderVertex};


use crate::{quad_font::{self, PixelSnapping}, ui_tree::UiNode, vk_ash_renderer::{vk_mesh_manager::GeometryAllocationID, vk_rgba_manager::RgbaAllocID}, MaterialAssetID, MyMaterial, MyMesh, RenderMeshDesc, RenderWorkTasker, ResourceAssetID};

// Runs a destroy function on an inner value when the variable is dropped
mod mortal {
    pub struct Mortal<'a, T> {
        inner: T,
        destroy_fn:  Option<Box<dyn FnOnce(&mut T) + 'a>>
    }

    impl<'a, T> Mortal<'a, T> {
        pub fn new(inner: T, destroy_fn: impl FnOnce(&mut T) + 'a) -> Self {
            Self { inner, destroy_fn: Some(Box::new(destroy_fn)) }
        }
    }

    impl<T> Drop for Mortal<'_, T> {
        fn drop(&mut self) {
            // println!("Dropping object of type {}", std::any::type_name::<T>());
            let d_fn = self.destroy_fn.take().unwrap();
            (d_fn)(&mut self.inner)
        }
    }

    impl<T> std::ops::Deref for Mortal<'_, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    impl<T> std::ops::DerefMut for Mortal<'_, T> {

        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.inner
        }
    }
}


unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;
    use std::ffi::CStr;

    let message = CStr::from_ptr((*p_callback_data).p_message);
    match flag {
        Flag::VERBOSE => println!("{:?} - {:?}", typ, message),
        Flag::INFO => println!("{:?} - {:?}", typ, message),
        Flag::WARNING => println!("{:?} - {:?}", typ, message),
        Flag::ERROR => eprintln!("{:?} - {:?}", typ, message),
        _ => {},
    }
    vk::FALSE
}

// pub fn create_swapchain()
pub const FRAMES_IN_FLIGHT: usize = 3;

pub type RenderResource = (ResourceAssetID, Vec<MyMesh>, Vec<MyMaterial>);

pub struct RenderFrameDesc {
    pub tick_num: usize,
    pub view_transform: glam::Mat4,
    pub projection_transform: glam::Mat4,
    pub mesh_instances: Vec<RenderMeshDesc>,
    pub ui_nodes: Vec<UiNode>,
}

pub fn run(
    shutdown_receiver: Receiver<()>,
    window_receiver: Receiver<winit::window::Window>,
    resource_receiver: Receiver<RenderResource>,
    render_frame_work: Arc<RenderWorkTasker>
) -> Result<(), Box<dyn std::error::Error>> {

    let window = Arc::new(window_receiver.recv().expect("Failed to receive initial window"));

    let entry = unsafe { ash::Entry::load_from("C:\\Windows\\System32\\vulkan-1.dll")? };

    // Create the vulkan instance
    let vk_instance = {

        let (ver_major, ver_minor) = match unsafe { entry.try_enumerate_instance_version()?} {
            Some(version) => (vk::api_version_major(version), vk::api_version_minor(version)),
            None => (1, 0),
        };
        if ver_major < 1 || ver_minor < 3 {
            return Err("Vulkan >=1.3 is not supported on this system".into());
        }

        let app_name = c"hello world";
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::make_api_version(0, ver_major, ver_minor, 0))
            .engine_name(app_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0));

        let mut exts: Vec<_> = 
            ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?
            .to_vec();
        exts.push(ash::ext::debug_utils::NAME.as_ptr());

        let available_layers = unsafe { ash::Entry::enumerate_instance_layer_properties(&entry)? };

        let val_layer_name = c"VK_LAYER_KHRONOS_validation";
        let has_validation_layer = available_layers.iter().any(|prop| 
            prop.layer_name_as_c_str().is_ok_and(|str| 
                str == val_layer_name 
            )
        );
        if has_validation_layer == false {
            return Err("Validation layer not available".into());
        }

        let enabled_layers = [val_layer_name.as_ptr()];
        let create_flags = vk::InstanceCreateFlags::default();
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&exts)
            .enabled_layer_names(&enabled_layers)
            .flags(create_flags);
        
        let inst = unsafe { entry.create_instance(&create_info, None)? };
        Mortal::new(inst, |i| unsafe { i.destroy_instance(None) })
    };

    // Enable debugging
    let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &vk_instance);
    let _debug_utils_messenger = {
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let messenger = unsafe { debug_utils.create_debug_utils_messenger(&create_info, None)? };
        Mortal::new(messenger, |m| unsafe { debug_utils.destroy_debug_utils_messenger(*m, None) })
    };

    let surface_khr = unsafe {
        ash_window::create_surface(
            &entry,
            &vk_instance,
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            None
        )?
    };
    let surface_inst = Mortal::new(
        ash::khr::surface::Instance::new(&entry, &vk_instance),
        |s| unsafe { s.destroy_surface(surface_khr, None) }
    );

    // select a physical device with suitable properties
    let (physical_device, queue_family_idx) = {
        unsafe { vk_instance.enumerate_physical_devices()? }
        .into_iter()
        .filter_map(|p_device| {
            let qf_props = unsafe { vk_instance.get_physical_device_queue_family_properties(p_device) };

            // find a queue family with the proper queue flags
            let desired_flags = vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER;
            let (qf_idx, _) = qf_props
                .iter()
                .enumerate()
                .find(|(_, prop)| prop.queue_flags.contains(desired_flags))?;

            // check if present is supported
            let present_supported = unsafe { surface_inst.get_physical_device_surface_support(p_device, qf_idx as _, surface_khr).ok()? };
            if present_supported == false {
                return None;
            }

            // check if extension properties are supported
            let ext_props = unsafe { vk_instance.enumerate_device_extension_properties(p_device).ok()? };

            let swapchain_ext_supported = 
                ext_props.iter()
                .any(|ext| ext.extension_name_as_c_str().is_ok_and(|name| name == ash::khr::swapchain::NAME));
            if swapchain_ext_supported == false {
                return None;
            }

            // check if device has available formats and present modes for the given surface
            let formats = unsafe { surface_inst.get_physical_device_surface_formats(p_device, surface_khr).ok()? };
            if formats.is_empty() {
                return None;
            }
            let present_modes = unsafe { surface_inst.get_physical_device_surface_present_modes(p_device, surface_khr).ok()? };
            if present_modes.is_empty() {
                return None;
            }

            // check if it has 1.3 features we want
            // TODO: check every single device feature
            // for example, need multi_draw_indirect, draw_indirect_first_instance
            // also query the multi draw indirect max count
            // also, add a return message that makes it clear why the device was rejected
            // 
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut features13);
            unsafe { vk_instance.get_physical_device_features2(p_device, &mut features); }

            if features.features.draw_indirect_first_instance == vk::FALSE
                || features.features.multi_draw_indirect == vk::FALSE {
                return None;
            }

            if features13.dynamic_rendering == vk::FALSE
                || features13.synchronization2 == vk::FALSE {
                return None;
            }

            return Some((p_device, qf_idx));
        })
        .next()
        .ok_or("No suitable physical device found")?
    };

    let surface_format = {
        let surface_formats = unsafe { surface_inst.get_physical_device_surface_formats(physical_device, surface_khr)? };
        if surface_formats.len() == 1 && surface_formats[0].format == vk::Format::UNDEFINED {
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            }
        } else {
            *surface_formats
                .iter()
                .find(|format| {
                    format.format == vk::Format::B8G8R8A8_UNORM
                        && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .unwrap_or(&surface_formats[0])
        }
    };

    // create a proper vulkan device
    let (vk_device, queue) = {
        // let queue_priorities = [1.0f32];
        let q_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_idx as _)
            .queue_priorities(&[1.0f32])];

        // let device_ext_ptrs
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true);
        let features = vk::PhysicalDeviceFeatures::default()
            .draw_indirect_first_instance(true)
            .multi_draw_indirect(true);
        let mut features_2 = vk::PhysicalDeviceFeatures2::default()
            .features(features)
            .push_next(&mut features13);

        let device_ext_ptrs = [ash::khr::swapchain::NAME.as_ptr()];

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&q_create_infos)
            .enabled_extension_names(&device_ext_ptrs)
            .push_next(&mut features_2);

        let device = unsafe { vk_instance.create_device(physical_device, &device_create_info, None)?};
        let m_device = Mortal::new(device, |d| unsafe { d.destroy_device(None) });
        let queue = unsafe { m_device.get_device_queue(queue_family_idx as _, 0) };
        (m_device, queue)
    };

    let vma_allocator = unsafe{ vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(&vk_instance, &vk_device, physical_device.clone()))?};

    let mut mesh_manager = vk_mesh_manager::MeshManager::new(&vma_allocator, &vk_device)?;
    let mut rgba_manager = vk_rgba_manager::RgbaManager::new(&vma_allocator, &vk_device)?;

    const MAX_GLOBAL_SKINNED_VERTICES: u32 = 1024 * 64;
    let global_skinned_render_vertices_allocation = mesh_manager.allocate::<RenderVertex>(MAX_GLOBAL_SKINNED_VERTICES as _)?;

    let command_pool = {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_idx as _)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cp = unsafe { vk_device.create_command_pool(&command_pool_info, None)? };

        Mortal::new(cp, |cp| unsafe { vk_device.destroy_command_pool(*cp, None) })
    };

    let depth_format = vk::Format::D32_SFLOAT;

    struct DescCount(u32);
    // each binding is a descriptor type, a descriptor count, and a shader stage
    let make_desc_set_layout = |bindings: &[(usize, vk::DescriptorType, DescCount, vk::ShaderStageFlags)]| 
    -> Result<_, Box<dyn std::error::Error>> {
        let bindings = bindings.iter().map(|(idx, ty, count, stage)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(*idx as _)
                .descriptor_type(*ty)
                .descriptor_count(count.0)
                .stage_flags(*stage)
        }).collect::<Vec<_>>();

        let desc_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);

        Ok(Mortal::new(
            unsafe { vk_device.create_descriptor_set_layout(&desc_set_layout_info, None)? },
            |d| unsafe { vk_device.destroy_descriptor_set_layout(*d, None) }
        ))
    };

    let apple_desc_set_layout = 
        make_desc_set_layout(&[
            (0, vk::DescriptorType::UNIFORM_BUFFER, DescCount(1), vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            (1, vk::DescriptorType::SAMPLER, DescCount(1), vk::ShaderStageFlags::FRAGMENT),
            (2, vk::DescriptorType::STORAGE_BUFFER, DescCount(1), vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            (3, vk::DescriptorType::SAMPLED_IMAGE, DescCount(vk_rgba_manager::MAX_TEXTURE_COUNT), vk::ShaderStageFlags::FRAGMENT),
            (4, vk::DescriptorType::STORAGE_BUFFER, DescCount(1), vk::ShaderStageFlags::FRAGMENT),
            (5, vk::DescriptorType::STORAGE_BUFFER, DescCount(1), vk::ShaderStageFlags::VERTEX),
        ])?;

    let skinny_desc_set_layout = 
        make_desc_set_layout(&[
            // storage buffer for vertices
            (0, vk::DescriptorType::STORAGE_BUFFER, DescCount(1), vk::ShaderStageFlags::COMPUTE),
            // storage buffer for buffer offsets
            (1, vk::DescriptorType::STORAGE_BUFFER, DescCount(1), vk::ShaderStageFlags::COMPUTE),
            // storage buffer for joint matrices
            (2, vk::DescriptorType::STORAGE_BUFFER, DescCount(1), vk::ShaderStageFlags::COMPUTE),
        ])?;


    let desc_pool  = {
        let mubo_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(FRAMES_IN_FLIGHT as _);

        let sampler_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(FRAMES_IN_FLIGHT as _);

        let flighted_storage_buffer_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(FRAMES_IN_FLIGHT as _);

        let tex_array_pool_size = rgba_manager.descriptor_pool_size(FRAMES_IN_FLIGHT as _);

        let pool_sizes = [
            mubo_pool_size, sampler_pool_size, flighted_storage_buffer_size, tex_array_pool_size,
            flighted_storage_buffer_size,
            flighted_storage_buffer_size, // vertex buffer for compute shader
            flighted_storage_buffer_size, // buffer offsets for compute shader
            flighted_storage_buffer_size, // joint matrices for compute shader
            flighted_storage_buffer_size, // letter objects for font rendering vertex shader
        ];

        let desc_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets((FRAMES_IN_FLIGHT as u32) * 2)
            .pool_sizes(&pool_sizes);

        Mortal::new(
            unsafe { vk_device.create_descriptor_pool(&desc_pool_info, None)? },
            |dp| unsafe { vk_device.destroy_descriptor_pool(*dp, None) }
        )
    };

    // descriptor sets allocated at the start
    let apple_desc_sets = {
        let layouts = 
            (0..FRAMES_IN_FLIGHT)
            .map(|_| *apple_desc_set_layout)
            .collect::<Vec<_>>();

        let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(*desc_pool)
            .set_layouts(&layouts);

        let sets = unsafe { vk_device.allocate_descriptor_sets(&desc_set_alloc_info)? };

        sets
    };

    let skinny_desc_sets = {
        let layouts = 
            (0..FRAMES_IN_FLIGHT)
            .map(|_| *skinny_desc_set_layout)
            .collect::<Vec<_>>();

        let desc_set_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(*desc_pool)
            .set_layouts(&layouts);

        let sets = unsafe { vk_device.allocate_descriptor_sets(&desc_set_alloc_info)? };

        sets
    };

    


    let read_shader = |bytes: &[u8]| -> Result<_, Box<dyn std::error::Error>> {
        let mut cursor = std::io::Cursor::new(bytes);
        let spirv_src = ash::util::read_spv(&mut cursor)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(&spirv_src);
        let module_res = unsafe { vk_device.create_shader_module(&create_info, None) };

        match module_res {
            Ok(module) => {
                let mortal = Mortal::new(module, |m| unsafe { vk_device.destroy_shader_module(*m, None) });
                return Ok(mortal);
            },
            Err(e) => Err(Box::new(e))
        }
    };

    // let shader_modules = {
    //     let shader_reg = [
    //         ("mesh_prim:unlit", include_bytes!("../shaders/spirv/mesh_prim_unlit.frag.spv").to_vec()),
    //         ("mesh_prim:standard", include_bytes!("../shaders/spirv/mesh_prim.frag.spv").to_vec()),
    //         ("mesh_prim:vert", include_bytes!("../shaders/spirv/mesh_prim.vert.spv").to_vec()),
    //         ("skinning", include_bytes!("../shaders/spirv/skinning.comp.spv").to_vec()),
    //         ("ui:font", include_bytes!("../shaders/spirv/ui_font.frag.spv").to_vec()),
    //         ("ui:quad", include_bytes!("../shaders/spirv/ui_quad.vert.spv").to_vec()),
    //         ("ui:solid_color", include_bytes!("../shaders/spirv/ui_solid_color.frag.spv").to_vec()),
    //     ];
    //     let mut shader_modules = HashMap::new();
    //     for (id, spirv_bytes) in shader_reg {
    //         let module = read_shader(&spirv_bytes)?;
    //         shader_modules.insert(id, module);
    //     }
    //     shader_modules
    // };

    let m_g_mesh_prim_pipeline_layout = Mortal::new(
        unsafe { 
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&apple_desc_set_layout));
            vk_device.create_pipeline_layout(&layout_info, None)? 
        }, 
        |pl| unsafe { vk_device.destroy_pipeline_layout(*pl, None) }
    );

    let m_g_mesh_prim_vert_shader_mod = read_shader(include_bytes!("../shaders/spirv/mesh_prim.vert.spv"))?;

    let g_mesh_prim_pipeline = {
        let entry_point_name = c"main";

        let frag_shader_mod = read_shader(include_bytes!("../shaders/spirv/mesh_prim.frag.spv"))?;
        let shader_states_info = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(*m_g_mesh_prim_vert_shader_mod)
                .name(&entry_point_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(*frag_shader_mod)
                .name(&entry_point_name),
        ];

        let (vert_binding_desc, vert_attr_descs) = RenderVertex::create_binding_information();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&vert_binding_desc))
            .vertex_attribute_descriptions(&vert_attr_descs);

        let input_asm_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let viewport_info = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0);

        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)];

        let color_blending_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let color_attachment_formats = [surface_format.format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(depth_format);
        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_compare_op(vk::CompareOp::LESS)
            .min_depth_bounds(0f32)
            .max_depth_bounds(1f32)
            .depth_bounds_test_enable(false)
            .depth_test_enable(true)
            .depth_write_enable(true);
            // .flags(vk::PipelineDepthStencilStateCreateFlags::RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_ARM);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_states_info)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_asm_info)
            .viewport_state(&viewport_info)
            .dynamic_state(&dynamic_state_create_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            .color_blend_state(&color_blending_info)
            .layout(m_g_mesh_prim_pipeline_layout.clone())
            .depth_stencil_state(&depth_stencil_state)
            .push_next(&mut rendering_info);
        
        let pipeline = unsafe {
            vk_device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| e.1)?[0]
        };

        let m_pipeline = Mortal::new(pipeline, |p| unsafe { vk_device.destroy_pipeline(*p, None) });

        m_pipeline
    };


    let make_ui_pipeline = |pc_range: vk::PushConstantRange, vert_spirv: &[u8], frag_spirv: &[u8]| 
    -> Result<_, Box<dyn std::error::Error>> {
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&apple_desc_set_layout))
            .push_constant_ranges(std::slice::from_ref(&pc_range));

        let m_pipeline_layout = Mortal::new(
            unsafe { vk_device.create_pipeline_layout(&layout_info, None)? }, 
            |pl| unsafe { vk_device.destroy_pipeline_layout(*pl, None) }
        );

        let entry_point_name = c"main";

        let vert_shader_mod = read_shader(vert_spirv)?;
        let frag_shader_mod = read_shader(frag_spirv)?;
        let shader_states_info = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(*vert_shader_mod)
                .name(&entry_point_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(*frag_shader_mod)
                .name(&entry_point_name),
        ];


        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let input_asm_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let viewport_info = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0);

        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)];

        let color_blending_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let color_attachment_formats = [surface_format.format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(depth_format);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_compare_op(vk::CompareOp::NEVER)
            .min_depth_bounds(0f32)
            .max_depth_bounds(1f32)
            .depth_bounds_test_enable(false)
            .depth_test_enable(false)
            .depth_write_enable(false);
            // .flags(vk::PipelineDepthStencilStateCreateFlags::RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_ARM);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_states_info)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_asm_info)
            .viewport_state(&viewport_info)
            .dynamic_state(&dynamic_state_create_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            .color_blend_state(&color_blending_info)
            .layout(m_pipeline_layout.clone())
            .depth_stencil_state(&depth_stencil_state)
            .push_next(&mut rendering_info);
        
        let pipeline = unsafe {
            vk_device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| e.1)?[0]
        };

        let m_pipeline = Mortal::new(pipeline, |p| unsafe { vk_device.destroy_pipeline(*p, None) });
        Ok((m_pipeline_layout, m_pipeline))
    };

    #[repr(C)]
    struct UIQuadJobPushConstant {
        color: [f32 ; 4],
        resolution: [f32 ; 4],
        glyph_texture_index: u32,
        padding: [u32 ; 3],
    }

    let ui_quad_push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<UIQuadJobPushConstant>() as _);

    let (m_g_font_ui_pipeline_layout, g_font_ui_pipeline) = make_ui_pipeline(
        ui_quad_push_constant_range.clone(),
        include_bytes!("../shaders/spirv/ui_quad.vert.spv"),
        include_bytes!("../shaders/spirv/ui_font.frag.spv"), 
    )?;

    let (m_g_solid_color_ui_pipeline_layout, g_solid_color_ui_pipeline) = make_ui_pipeline(
        ui_quad_push_constant_range.clone(),
        include_bytes!("../shaders/spirv/ui_quad.vert.spv"),
        include_bytes!("../shaders/spirv/ui_solid_color.frag.spv"), 
    )?;


    let (c_skin_pipeline_layout, c_skin_pipeline) = {

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<SkinningJobBufferObj>() as _);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&skinny_desc_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let m_pipeline_layout = Mortal::new(
            unsafe { vk_device.create_pipeline_layout(&layout_info, None)? }, 
            |pl| unsafe { vk_device.destroy_pipeline_layout(*pl, None) }
        );

        let entry_point_name = c"main";

        let compute_shader_mod = read_shader(include_bytes!("../shaders/spirv/skinning.comp.spv"))?;
        let shader_states_info = 
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(*compute_shader_mod)
                .name(&entry_point_name);

        let create_infos = [
            vk::ComputePipelineCreateInfo::default()
            .layout(*m_pipeline_layout)
            .stage(shader_states_info)
        ];

        let pipeline = unsafe { 
            vk_device.create_compute_pipelines(vk::PipelineCache::null(), &create_infos, None)
            .map_err(|e| e.1)?[0] 
        };
        let m_pipeline = Mortal::new(pipeline, |p| unsafe { vk_device.destroy_pipeline(*p, None) });

        (m_pipeline_layout, m_pipeline)
    };

    let swapchain = ash::khr::swapchain::Device::new(&vk_instance, &vk_device);
    let create_swapchain_khr = || -> Result<_, Box<dyn std::error::Error>> {
        // TODO: rewrite this to make sense
        // Swapchain format
            
        // Swapchain present mode
        let present_mode = {
            let present_modes = unsafe { surface_inst.get_physical_device_surface_present_modes(physical_device, surface_khr)? };
            if present_modes.contains(&vk::PresentModeKHR::FIFO) {
                vk::PresentModeKHR::FIFO
            } else {
                vk::PresentModeKHR::IMMEDIATE
            }
        };
    
        let capabilities = unsafe { surface_inst.get_physical_device_surface_capabilities(physical_device, surface_khr)? };
        let supports_transfer = capabilities.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_DST);
        if supports_transfer == false {
            return Err("Surface does not support transfer".into());
        }

        // Swapchain extent
        let extent = {
            if capabilities.current_extent.width != std::u32::MAX {
                capabilities.current_extent
            } else {
                let min = capabilities.min_image_extent;
                let max = capabilities.max_image_extent;
                let win_size = window.inner_size();
                let width = win_size.width.min(max.width).max(min.width);
                let height = win_size.height.min(max.height).max(min.height);
                vk::Extent2D { width, height }
            }
        };
    
        // Swapchain image count
        if capabilities.min_image_count > FRAMES_IN_FLIGHT as _ || capabilities.max_image_count < FRAMES_IN_FLIGHT as _ {
            return Err("Frames in flight count not supported".into());
        }
    
        // Swapchain
        let create_info = {
            let builder = vk::SwapchainCreateInfoKHR::default()
                .surface(surface_khr)
                .min_image_count(FRAMES_IN_FLIGHT as _)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE);
    
            builder
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };
    
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None)? };
        let m_swapchain_khr = Mortal::new(swapchain_khr, |sck| unsafe { swapchain.destroy_swapchain(*sck, None) });
    
        // Swapchain images and image views
        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr)? };
        let views = images
            .iter()
            .map(|image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
    
                unsafe { vk_device.create_image_view(&create_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let m_views = Mortal::new(views, |views| {
            views.iter().for_each(|view| unsafe { vk_device.destroy_image_view(*view, None) })
        });

        return Ok((m_swapchain_khr, extent, images, m_views));
    };
    let (swapchain_khr, sc_extent, sc_images, sc_views) = create_swapchain_khr()?;

    let find_memory_type = |type_filter : u32, req_properties: vk::MemoryPropertyFlags|
    -> Result<u32, Box<dyn std::error::Error>>{
        let mem_props = unsafe { vk_instance.get_physical_device_memory_properties(physical_device) };

        for idx in 0..mem_props.memory_type_count {
            let i_prop_flags = mem_props.memory_types[idx as usize].property_flags;
            if type_filter & (1 << idx) != 0 && i_prop_flags.contains(req_properties) {
                return Ok(idx);
            }
        }

        return Err("Failed to find suitable memory type".into());
    };

    let dumb_allocate_mem = |mem_reqs : vk::MemoryRequirements, flags : vk::MemoryPropertyFlags|
    -> Result<_, Box<dyn std::error::Error>> {
        // let buffer_mem_reqs = unsafe { vk_device.get_buffer_memory_requirements(*buffer) };

        let mem_type = find_memory_type(mem_reqs.memory_type_bits, flags)?;

        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);

        Ok(Mortal::new(
            unsafe { vk_device.allocate_memory(&info, None)? },
            |m| unsafe { vk_device.free_memory(*m, None) }
        ))
    };

    let dumb_buffer_create = |size: _, usage: _, flags: _| 
    -> Result<_, Box<dyn std::error::Error>> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = Mortal::new(
            unsafe { vk_device.create_buffer(&buffer_info, None)? },
            |b| unsafe { vk_device.destroy_buffer(*b, None) }
        );

        let mem = dumb_allocate_mem(
            unsafe { vk_device.get_buffer_memory_requirements(*buffer) },
            flags
        )?;

        unsafe { vk_device.bind_buffer_memory(*buffer, *mem, 0)? };

        Ok((mem, buffer))
    };



    #[repr(C)]
    struct MyUniformBufferObj {
        model: glam::Mat4,
        view: glam::Mat4,
        proj: glam::Mat4
    }

    
    let (_mubo_buffer_mems, mubo_buffers, mubo_pointers) = {

        let struct_size = std::mem::size_of::<MyUniformBufferObj>() as u64;

        let mut buffers = Vec::new();
        let mut mems = Vec::new();
        let mut ptrs = Vec::new();

        for _ in 0..FRAMES_IN_FLIGHT {
            let (mem, buffer) = dumb_buffer_create(
                struct_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER, 
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            )?;
            let dst_ptr = unsafe { vk_device.map_memory(*mem, 0, struct_size, vk::MemoryMapFlags::empty())? };
            mems.push(mem);
            buffers.push(buffer);
            ptrs.push(dst_ptr as *mut u8);
        }

        (mems, buffers, ptrs)
    };

    #[repr(C)]
    struct RenderInstanceBufferObj {
        model_mat: glam::Mat4,
        texture_group_index: u32,
        padding: [u32 ; 3]
    }

    #[repr(C)]
    struct UIQuadBufferObj {
        base_pos_size: [f32 ; 4],
        start_end_tex_coords: [f32 ; 4],
    }

    let mut render_instance_buffers = vk_slab_manager::DynamicFlightedArray::<RenderInstanceBufferObj, FRAMES_IN_FLIGHT>::new(
            300, &vma_allocator, &vk_device, 
            vk::BufferUsageFlags::STORAGE_BUFFER)?;

    let mut indirect_buffer = vk_slab_manager::DynamicFlightedArray::<vk::DrawIndexedIndirectCommand, FRAMES_IN_FLIGHT>::new(
        300, &vma_allocator, &vk_device, 
        vk::BufferUsageFlags::INDIRECT_BUFFER
    )?;

    let mut skinning_joint_matrices = vk_slab_manager::DynamicFlightedArray::<glam::Mat4, FRAMES_IN_FLIGHT>::new(
        1024, &vma_allocator, &vk_device, 
        vk::BufferUsageFlags::STORAGE_BUFFER
    )?;

    let mut ui_quad_buffers = vk_slab_manager::DynamicFlightedArray::<UIQuadBufferObj, FRAMES_IN_FLIGHT>::new(
        1024 * 8, &vma_allocator, &vk_device, 
        vk::BufferUsageFlags::STORAGE_BUFFER
    )?;

    #[derive(Debug)]
    #[repr(C)]
    struct SkinningJobBufferObj {
        bone_vertices_index: u32,
        joint_transforms_index: u32,
        base_render_vertices_index: u32,
        output_render_vertices_index: u32,
        vertex_count: u32,
    }

    let skin_instance_buffers = vk_slab_manager::DynamicFlightedArray::<SkinningJobBufferObj, FRAMES_IN_FLIGHT>::new(
        300, &vma_allocator, &vk_device, 
        vk::BufferUsageFlags::STORAGE_BUFFER
    )?;

    // Create Depth Image
    let (_depth_img_mem, depth_img, depth_img_view) = {
        // TODO: determine if we should check for depth format support
        // and tiling optimal support
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: sc_extent.width,
                height: sc_extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(depth_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let img = Mortal::new(
            unsafe { vk_device.create_image(&create_info, None)? },
            |i| unsafe { vk_device.destroy_image(*i, None) }
        );
        
        let memory = dumb_allocate_mem(
            unsafe { vk_device.get_image_memory_requirements(*img) },
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        )?;

        unsafe { vk_device.bind_image_memory(*img, *memory, 0)? };

        let iv_create_info = vk::ImageViewCreateInfo::default()
            .image(*img)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let iv = Mortal::new(
            unsafe { vk_device.create_image_view(&iv_create_info, None)? },
            |iv| unsafe { vk_device.destroy_image_view(*iv, None) }
        );

        (memory, img, iv)
    };


    // Dumb utility functions
    let dumb_transition_img_layout_barrier = |img: &vk::Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, array_layer: u32| {
        let img_mem_barrier = vk::ImageMemoryBarrier2::default()
            .image(*img)
            // TODO: make the stage masks more robust
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .subresource_range(vk::ImageSubresourceRange {
                base_array_layer: array_layer,
                aspect_mask: vk::ImageAspectFlags::COLOR,
                layer_count: 1,
                level_count: 1,
                ..Default::default()
            });
        return img_mem_barrier;
    };

    let dumb_mem_to_stage_buffer = |d_size: vk::DeviceSize, src_ptr: *const u8|
    -> Result<_, Box<dyn std::error::Error>> {

        let (buffer_mem, buffer) = dumb_buffer_create(
            d_size, vk::BufferUsageFlags::TRANSFER_SRC, 
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        )?;

        let map_ptr = unsafe { vk_device.map_memory(*buffer_mem, 0, d_size, vk::MemoryMapFlags::empty())? as *mut u8 };
        unsafe { std::ptr::copy(src_ptr, map_ptr, d_size as usize) };
        unsafe { vk_device.unmap_memory(*buffer_mem) };

        return Ok((buffer_mem, buffer))
    };

    let dumb_run_cmd_once = |cmd_mod: &dyn Fn(&vk::CommandBuffer)| 
    -> Result<(), Box<dyn std::error::Error>> {
        let upload_fence = {
            let fence_info = vk::FenceCreateInfo::default();
            Mortal::new(
                unsafe { vk_device.create_fence(&fence_info, None)? },
                |s| unsafe { vk_device.destroy_fence(*s, None) }
            )
        };

        let cmd = {
            let cmd_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(*command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let cmd = Mortal::new(
                unsafe { vk_device.allocate_command_buffers(&cmd_info)?[0] },
                |c| unsafe { vk_device.free_command_buffers(*command_pool, std::slice::from_ref(c)) }
            );

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { vk_device.begin_command_buffer(*cmd, &begin_info)? };

            cmd_mod(&cmd);

            unsafe {vk_device.end_command_buffer(*cmd)?}

            cmd
        };

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cmd));

        unsafe { vk_device.queue_submit(queue, std::slice::from_ref(&submit_info), *upload_fence)? }

        let long_duration = 100_000_000;
        unsafe { vk_device.wait_for_fences(std::slice::from_ref(&upload_fence), true, long_duration)? }

        return Ok(());
    };

    let dumb_mem_to_specific_device_buffer =
    |d_size: vk::DeviceSize, src_ptr: *const u8, dst_buffer: &vk::Buffer, dst_offset: vk::DeviceSize|
    -> Result<(), Box<dyn std::error::Error>> {
        let total_size = d_size as u64;

        let (_stag_buffer_mem, stag_buffer) = dumb_mem_to_stage_buffer(total_size, src_ptr as _)?;

        let cmd_mod = |cmd: &vk::CommandBuffer| {
            let copy_region = vk::BufferCopy::default()
                .size(total_size)
                .src_offset(0)
                .dst_offset(dst_offset);

            unsafe { vk_device.cmd_copy_buffer(*cmd, *stag_buffer, *dst_buffer, std::slice::from_ref(&copy_region)) };
        };

        dumb_run_cmd_once(&cmd_mod)?;

        Ok(())
    };

    struct RenderDeviceMeshPrimitive {
        render_geometry: IndexedVerticesAllocation,
        opt_material_id: Option<MaterialAssetID>,
        opt_skin_geometry_id: Option<GeometryAllocationID>,
    }

    struct RenderDeviceMesh {
        primitives: Vec<RenderDeviceMeshPrimitive>,
    }

    let dumb_img_upload = |rgba_man: &mut vk_rgba_manager::RgbaManager, data: &Vec<u8>, img_id: RgbaAllocID| 
    -> Result<(), Box<dyn std::error::Error>>{
        let (_buffer_mem, buffer) = dumb_mem_to_stage_buffer(data.len() as u64, data.as_ptr())?;

        // TODO make this more safer without unwrap
        // TODO: transition image layout somehow automatically
        // let rgba_view = rgba_man.get_view(img_id).unwrap();
        let (img, layer_index) = rgba_man.vk_image_information(img_id).unwrap();

        let cmd_mod = |cmd: &vk::CommandBuffer| {
            {
                let barrier = dumb_transition_img_layout_barrier(&img, 
                    vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    layer_index
                );
                let dep = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier));

                unsafe { vk_device.cmd_pipeline_barrier2(*cmd, &dep) };
            }

            let copy_region = rgba_man.buffer_to_image_info(img_id);

            unsafe { vk_device.cmd_copy_buffer_to_image(*cmd, *buffer, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, std::slice::from_ref(&copy_region)) };
                
            {
                let barrier = dumb_transition_img_layout_barrier(&img, 
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, 
                    layer_index
                );
                let dep = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier));

                unsafe { vk_device.cmd_pipeline_barrier2(*cmd, &dep) };
            }
        };

        dumb_run_cmd_once(&cmd_mod)?;

        Ok(())
    };

    let mut material_manager = MaterialManager::new(&vma_allocator, &vk_device)?;

    // setup font
    const ATLAS_PIXEL_WIDTH: u32 = 2048;
    const FONT_TEXEL_SIZE: u32 = 64;
    let (quad_font, q_font_img_id) = {
        let font = fontsdf::Font::from_bytes(include_bytes!("../assets/fonts/DejaVuSansMono-Regular.ttf")).unwrap();

        let q_font = quad_font::QuadFont::new(font, FONT_TEXEL_SIZE, ATLAS_PIXEL_WIDTH);
        let img_id = rgba_manager.allocate(PowerTwoTextureLength::L2048)?;
        dumb_img_upload(&mut rgba_manager, &q_font.atlas, img_id)?;
        (q_font, img_id)
    };
    println!("quad font image id: {:?}", q_font_img_id);

    struct RenderPrimitiveJob {
        opt_texture_group_index: Option<usize>,
        vertices_view: GeometryAllocationView<RenderVertex>,
        indices_view: GeometryAllocationView<u32>,
        model_transform: glam::Mat4,
    }

    let tex_sampler = {
        // TODO: add anisotropy
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .max_anisotropy(1f32)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0f32)
            .min_lod(0f32)
            .max_lod(0f32);

        Mortal::new(
            unsafe { vk_device.create_sampler(&sampler_info, None)? },
            |s| unsafe { vk_device.destroy_sampler(*s, None) }
        )
    };

    let prim_command_buffers = {
        let cba_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as _);

        unsafe { vk_device.allocate_command_buffers(&cba_info)? }
    };


    let image_available_semaphore = {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let sem = unsafe { vk_device.create_semaphore(&semaphore_info, None)? };
        Mortal::new(sem, |s| unsafe { vk_device.destroy_semaphore(*s, None) })
    };
    let render_finished_semaphore = {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let sem = unsafe { vk_device.create_semaphore(&semaphore_info, None)? };
        Mortal::new(sem, |s| unsafe { vk_device.destroy_semaphore(*s, None) })
    };
    let fence = {
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { vk_device.create_fence(&fence_info, None)? };
        Mortal::new(fence, |s| unsafe { vk_device.destroy_fence(*s, None) })
    };

    let mut loaded_mesh_registry = HashMap::new();

    let mut loaded_resources = HashSet::<ResourceAssetID>::new();

    let mut render_iteration = 0;
    loop {
        
        match shutdown_receiver.try_recv() {
            Ok(_) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                break;
            },
            _ => {}
        }

        // check if work is available
        let render_frame_desc = {
            if let Some(frame_work) = render_frame_work.try_get_work() {
                frame_work
            } else {
                // no work, skip this iteration
                // TODO: don't do busy waiting
                continue;
            }
        };

        let required_resources = {
            let mut req_resources = HashSet::new();
            for mesh_inst in render_frame_desc.mesh_instances.iter() {
                let id = mesh_inst.mesh_asset_id.gltf.resource_id;
                req_resources.insert(id);
            }

            req_resources
        };

        // load in resources
        for (res_id, meshes, materials) in resource_receiver.try_iter() {
            println!("received resource id: {:?}", res_id);

            // load materials
            for material in materials {
                let upload_jobs = material_manager.add_material(material, &mut rgba_manager)?;
                for (data, alloc_id) in upload_jobs {
                    dumb_img_upload(&mut rgba_manager, &data, alloc_id)?;
                }
            }
    
            // load meshes
            for mesh in meshes {
                let mesh_id = mesh.self_id;
                println!("received mesh: {:?}", mesh_id);
                let mut internal_prims = Vec::new();
                for primitive in mesh.primitives {
                    let (render_alloc, jobs) = mesh_manager.allocate_primitive(&primitive)?;

                    // upload data to render allocation
                    for job in jobs {
                        dumb_mem_to_specific_device_buffer(
                            job.data.len() as u64,
                            job.data.as_ptr(),
                            mesh_manager.buffer(),
                            job.buffer_byte_offset,
                        )?;
                    }

                    // upload skinning geometry to GPU if present
                    let opt_skin_geometry_id =
                        if let Some(joints) = primitive.opt_joints {
                            // make skinning geometry
                            let mut bone_vertices = Vec::new();
                            for (i, joint) in joints.iter().enumerate() {
                                let prim_pos = primitive.positions[i];
                                let pos = [prim_pos[0], prim_pos[1], prim_pos[2], 0.0];
                                let b_vertex = BoneVertex {
                                    pos,
                                    joint_indices: [
                                        joint.indices[0] as u32,
                                        joint.indices[1] as u32,
                                        joint.indices[2] as u32,
                                        joint.indices[3] as u32,
                                    ],
                                    joint_weights: joint.weights,
                                };
                                bone_vertices.push(b_vertex);
                            }

                            // allocate and upload vertices
                            let bone_vertices_allocation = mesh_manager.allocate::<BoneVertex>(bone_vertices.len() as u64)?;
                            let view = mesh_manager.get_view::<BoneVertex>(bone_vertices_allocation).unwrap();
                            dumb_mem_to_specific_device_buffer(
                                view.byte_size,
                                bone_vertices.as_ptr() as *const u8,
                                mesh_manager.buffer(),
                                view.buffer_byte_offset,
                            )?;
                            Some(bone_vertices_allocation)
                        } else {
                            None
                        };
    
                    internal_prims.push(RenderDeviceMeshPrimitive {
                        render_geometry: render_alloc,
                        opt_material_id: primitive.material_id,
                        opt_skin_geometry_id,
                    });
                }
    
                let rd_mesh = RenderDeviceMesh {
                    primitives: internal_prims,
                };
    
                loaded_mesh_registry.insert(mesh_id, rd_mesh);
            }
    

            loaded_resources.insert(res_id);
        };

        if required_resources.is_subset(&loaded_resources) == false {
            println!("resource not present");
            continue;
        }

        unsafe { vk_device.wait_for_fences(&[*fence], true, std::u64::MAX)? };

        // Drawing the frame
        if render_iteration % 100 == 0 {
            // println!("render iteration: {}", render_iteration);
        }
        render_iteration += 1;

        let next_image_result = unsafe {
            swapchain.acquire_next_image(
                *swapchain_khr,
                std::u64::MAX,
                *image_available_semaphore,
                vk::Fence::null(),
            )
        };
        let image_index = match next_image_result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // println!("Out of date");
                continue; // go to next frame
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { vk_device.reset_fences(&[fence.clone()])? };

        let wait_semaphore_submit_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(image_available_semaphore.clone())
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);

        let signal_semaphore_submit_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished_semaphore.clone())
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);


        let mut render_prim_work: Vec<RenderPrimitiveJob> = Vec::new();
        struct SkinningJob {
            base_render_geometry: GeometryAllocationView<RenderVertex>,
            output_render_geometry: GeometryAllocationView<RenderVertex>,
            skinning_geometry: GeometryAllocationID,
            joint_transforms: Vec<glam::Mat4>,
        }
        let mut skinning_work = Vec::new();

        // get global skinned vertices allocation view
        let global_skinned_verts = mesh_manager.get_view::<RenderVertex>(global_skinned_render_vertices_allocation).unwrap();
        let mut skinned_verts_used = 0;
        // let suballocated_skinned_verts: Vec<GeometryAllocationView<MyRenderVertex>> = Vec::new();
        // global_skinned_verts.

        // setup render primitive instances
        for mesh_desc in render_frame_desc.mesh_instances.iter() {
            // todo: wait for resources
            let mesh = loaded_mesh_registry.get(&mesh_desc.mesh_asset_id).unwrap();
            for primitive in mesh.primitives.iter() {
                let opt_texture_group_index = 
                    primitive.opt_material_id.as_ref()
                    .and_then(|id| material_manager.texture_group_index(id));

                let indices_view = mesh_manager.get_view::<u32>(primitive.render_geometry.index_allocation_id).unwrap();
                let base_render_vertices_view = mesh_manager.get_view::<RenderVertex>(primitive.render_geometry.vertex_allocation_id).unwrap();

                // the allocation of vertices to render
                let render_vertices_view;

                if let Some(joint_transforms) = &mesh_desc.opt_joint_transforms {
                    // add skinning job if skinning requested

                    // TODO: guarantee that skinning geometry is present in a mesh with joint transform
                    let skin_geometry = primitive.opt_skin_geometry_id.unwrap();

                    // skinned mesh, use suballocation of the global skinned mesh allocation
                    let vertex_count = base_render_vertices_view.get_element_count();
                    render_vertices_view = global_skinned_verts.get_sub_view(skinned_verts_used, vertex_count);
                    skinned_verts_used += vertex_count;

                    let job = SkinningJob {
                        base_render_geometry: base_render_vertices_view,
                        output_render_geometry: render_vertices_view.clone(),
                        skinning_geometry: skin_geometry,
                        joint_transforms: joint_transforms.clone(),
                    };
                    skinning_work.push(job);
                } else {
                    render_vertices_view = base_render_vertices_view;
                }

                let obj = RenderPrimitiveJob {
                    model_transform: mesh_desc.model_transform,
                    opt_texture_group_index,
                    indices_view,
                    vertices_view: render_vertices_view,
                };
                render_prim_work.push(obj);
            }
        }

        // update render instance buffer
        {
            let slice = render_instance_buffers.slice(image_index as _);
            for i in 0..render_prim_work.len() {
                let ri = &render_prim_work[i];
                let default_texture_group_index = 0;
                // TODO: provide a better default texture group index
                let data = RenderInstanceBufferObj {
                    model_mat: ri.model_transform,
                    texture_group_index: ri.opt_texture_group_index.unwrap_or(default_texture_group_index) as u32,
                    padding: [0 ; 3]
                };

                slice[i] = data;
            }
        }

        // update uniform buffer
        {
            let dst_ptr = mubo_pointers[image_index as usize];

            // todo: replace this with push constant
            let mubo = MyUniformBufferObj {
                model: glam::Mat4::IDENTITY, // todo remove this
                view: render_frame_desc.view_transform,
                proj: render_frame_desc.projection_transform,
            };
            let data_ptr = std::ptr::addr_of!(mubo) as *const u8;
            unsafe { std::ptr::copy(data_ptr, dst_ptr, std::mem::size_of::<MyUniformBufferObj>()) };
        }

        enum UiQuadRenderType {
            Glyphs,
            SolidColor
        }

        struct UiQuadJob {
            render_type: UiQuadRenderType,
            color: [f32 ; 4],
            quad_start_index: u32,
            quad_count: u32,
        }

        let mut ui_quad_jobs = Vec::<UiQuadJob>::new();

        // update ui quad buffer
        {
            let slice = ui_quad_buffers.slice(image_index as _);
            let mut quads_added = 0;
            for ui_node in render_frame_desc.ui_nodes {
                let quad_start_index = quads_added as u32;
                match ui_node.content {
                    crate::ui_tree::UiContent::Text { text_str, color, font_size } => {
                        let quad_start_index = quads_added as u32;

                        const GLYPH_WIDTH_UI_SPACE: f32 = 17.0 * 0.001;
                        let aspect_ratio = sc_extent.width as f32 / sc_extent.height as f32;
                        let glyph_height_ui_space: f32 = GLYPH_WIDTH_UI_SPACE * aspect_ratio;
                        const QUAD_WIDTH_TO_U_WIDTH: f32 =  (FONT_TEXEL_SIZE as f32) / (GLYPH_WIDTH_UI_SPACE as f32 * ATLAS_PIXEL_WIDTH as f32);
                        let glyph_width_to_v_width: f32 =  (FONT_TEXEL_SIZE as f32) / (glyph_height_ui_space as f32 * ATLAS_PIXEL_WIDTH as f32);

                        let pixel_snapping = PixelSnapping {
                            pixel_width: sc_extent.width,
                            pixel_height: sc_extent.height
                        };

                        let glyph_quads = quad_font::generate_quads(&quad_font, text_str.clone(), 
                            [ui_node.spacing.x, ui_node.spacing.y], [ui_node.spacing.width, ui_node.spacing.height], 
                            GLYPH_WIDTH_UI_SPACE, glyph_height_ui_space, quad_font::TextVerticalAlignment::Bottom,
                            Some(pixel_snapping));

                        for g_quad in glyph_quads.iter() {
                            // println!("quad width to u width: {}", QUAD_WIDTH_TO_U_WIDTH);
                            let u_end = g_quad.atlas_u + QUAD_WIDTH_TO_U_WIDTH * g_quad.width;
                            let v_end = g_quad.atlas_v + glyph_width_to_v_width * g_quad.height;
                            slice[quads_added] = UIQuadBufferObj {
                                base_pos_size: [g_quad.x, g_quad.y, g_quad.width, g_quad.height],
                                start_end_tex_coords: [g_quad.atlas_u, g_quad.atlas_v, u_end, v_end]
                            };

                            quads_added += 1;
                        }
                        ui_quad_jobs.push(UiQuadJob {
                            color,
                            render_type: UiQuadRenderType::Glyphs,
                            quad_start_index,
                            quad_count: glyph_quads.len() as u32,
                        });
                    },
                    crate::ui_tree::UiContent::FilledColor { color } => {
                        let quad_count = 1;
                        slice[quads_added] = UIQuadBufferObj {
                            base_pos_size: [ui_node.spacing.x, ui_node.spacing.y, ui_node.spacing.width, ui_node.spacing.height],
                            start_end_tex_coords: [0.0, 0.0, 1.0, 1.0],
                        };
                        quads_added += 1;

                        ui_quad_jobs.push(UiQuadJob {
                            color: color.clone(),
                            render_type: UiQuadRenderType::SolidColor,
                            quad_start_index,
                            quad_count,
                        });
                    },
                }
            }
        }

        // record command buffer
        {
            let cmd_buffer = prim_command_buffers[image_index as usize];
            unsafe { vk_device.reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())? };
            let cbb_info = vk::CommandBufferBeginInfo::default();
            unsafe { vk_device.begin_command_buffer(cmd_buffer, &cbb_info)? };

            // apply compute skinning to command buffer
            {
                // update compute descriptor set
                {
                    let vertex_buffer_info =
                        vk::DescriptorBufferInfo::default()
                            .buffer(*mesh_manager.buffer())
                            .offset(0)
                            .range(vk::WHOLE_SIZE);
                    // TODO: remove skinning instance buffer
                    let skinning_instance_buffer_info = 
                        vk::DescriptorBufferInfo::default()
                            .buffer(*skin_instance_buffers.buffer(image_index as usize))
                            .offset(0)
                            .range(vk::WHOLE_SIZE);
                    let joint_matrices_buffer_info = 
                        vk::DescriptorBufferInfo::default()
                            .buffer(*skinning_joint_matrices.buffer(image_index as usize))
                            .offset(0)
                            .range(vk::WHOLE_SIZE);
                    let write_infos = [
                        vk::WriteDescriptorSet::default()
                            .dst_set(skinny_desc_sets[image_index as usize])
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(std::slice::from_ref(&vertex_buffer_info)),
                        vk::WriteDescriptorSet::default()
                            .dst_set(skinny_desc_sets[image_index as usize])
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(std::slice::from_ref(&skinning_instance_buffer_info)),
                        vk::WriteDescriptorSet::default()
                            .dst_set(skinny_desc_sets[image_index as usize])
                            .dst_binding(2)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(std::slice::from_ref(&joint_matrices_buffer_info)),
                    ];

                    unsafe { vk_device.update_descriptor_sets(&write_infos, &[]) };
                }

                unsafe { vk_device.cmd_bind_descriptor_sets(cmd_buffer, vk::PipelineBindPoint::COMPUTE, *c_skin_pipeline_layout, 0, 
                    std::slice::from_ref(skinny_desc_sets.get(image_index as usize).unwrap()), &[]); }

                unsafe { vk_device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, *c_skin_pipeline); }

                // barrier, syncronize the vertex buffer from vertex to compute
                {
                    let vertex_buffer_barrier = vk::BufferMemoryBarrier2::default()
                        .buffer(*mesh_manager.buffer())
                        .size(vk::WHOLE_SIZE) // TODO: subbuffer syncronization
                        .src_access_mask(vk::AccessFlags2::MEMORY_READ)
                        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
                        .src_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
                    let dependency_info = vk::DependencyInfo::default()
                        .buffer_memory_barriers(std::slice::from_ref(&vertex_buffer_barrier));

                    unsafe { vk_device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info) };
                }

                // run dispatches with push constants
                {
                    let joint_matrices_slice = skinning_joint_matrices.slice(image_index as usize);
                    let mut jms_added = 0;


                    const SWS: u32 = 256; // Skinning Workgroup Size
                    for job in skinning_work.iter() {
                        // add joint matrices from job
                        let job_joint_matrix_index = jms_added; // the starting index of the joint matrices for this job
                        for joint_transform in job.joint_transforms.iter() {
                            joint_matrices_slice[jms_added] = *joint_transform;
                            jms_added += 1;
                        }

                        // create job push constant
                        let bone_verts_view = mesh_manager.get_view::<BoneVertex>(job.skinning_geometry).unwrap();
                        // let render_verts_view = mesh_manager.get_view::<MyRenderVertex>(job.base_render_geometry).unwrap();
                        let vertex_count = job.base_render_geometry.get_element_count() as u32;
                        // println!("vertex count job geometry: {}", vertex_count);

                        let pc_job = SkinningJobBufferObj {
                            bone_vertices_index: bone_verts_view.buffer_index_offset() as u32,
                            joint_transforms_index: job_joint_matrix_index as u32,
                            base_render_vertices_index: job.base_render_geometry.buffer_index_offset() as u32,
                            output_render_vertices_index: job.output_render_geometry.buffer_index_offset() as u32,
                            vertex_count,
                        };
                        // println!("job: {:?}", pc_job);
                        let pc_bytes = unsafe { std::slice::from_raw_parts(&pc_job as *const SkinningJobBufferObj as *const u8, std::mem::size_of::<SkinningJobBufferObj>()) };
                        unsafe { vk_device.cmd_push_constants(cmd_buffer, *c_skin_pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes); }

                        // run disptach job
                        unsafe { vk_device.cmd_dispatch(cmd_buffer, (vertex_count + (SWS - 1)) / SWS, 1, 1); }
                    }

                }

                // barrier, syncronize the vertex buffer from compute to vertex
                {
                    let vertex_buffer_barrier = vk::BufferMemoryBarrier2::default()
                        .buffer(*mesh_manager.buffer())
                        .size(vk::WHOLE_SIZE)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
                        .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT);
                    let dependency_info = vk::DependencyInfo::default()
                        .buffer_memory_barriers(std::slice::from_ref(&vertex_buffer_barrier));

                    unsafe { vk_device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info) };
                }
            }

            // general rendering
            {

                // memory transition layout for color and depth attachments
                {
                    let color_mem_barrier_one = vk::ImageMemoryBarrier2::default()
                        .image(sc_images[image_index as usize])
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            layer_count: 1,
                            level_count: 1,
                            ..Default::default()
                        });

                    let depth_stage_masks = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
                    let depth_mem_barrier_one = vk::ImageMemoryBarrier2::default()
                        .image(*depth_img)
                        .src_stage_mask(depth_stage_masks)
                        .dst_stage_mask(depth_stage_masks)
                        .src_access_mask(vk::AccessFlags2::empty())
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            layer_count: 1,
                            level_count: 1,
                            ..Default::default()
                        });

                    let mem_barriers_one = [color_mem_barrier_one, depth_mem_barrier_one];

                    let dependency_info = vk::DependencyInfo::default()
                        .image_memory_barriers(&mem_barriers_one);

                    unsafe { vk_device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info) };
                }

                // viewport scissors setup
                {
                    let viewports = [vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: sc_extent.width as _,
                        height: sc_extent.height as _,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }];
                    unsafe { vk_device.cmd_set_viewport(cmd_buffer, 0, &viewports) }

                    let scissors = [vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: (vk::Extent2D { 
                            width: sc_extent.width, 
                            height: sc_extent.height
                        }),
                    }];
                    unsafe { vk_device.cmd_set_scissor(cmd_buffer, 0, &scissors) }
                }


                // assign flighted buffers to descriptor sets
                {
                    let i = image_index as usize;
                    let sets = &apple_desc_sets;

                    let buffer_info = vk::DescriptorBufferInfo::default()
                        .buffer(*mubo_buffers[i])
                        .offset(0)
                        .range(std::mem::size_of::<MyUniformBufferObj>() as u64);

                    let image_info = vk::DescriptorImageInfo::default()
                        .sampler(*tex_sampler);

                    let buffer_write_info = vk::WriteDescriptorSet::default()
                        .dst_set(sets[i])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(std::slice::from_ref(&buffer_info));

                    let image_write_info = vk::WriteDescriptorSet::default()
                        .dst_set(sets[i])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .descriptor_count(1)
                        .image_info(std::slice::from_ref(&image_info));

                    let ssbo_info = render_instance_buffers.buffer_descriptor(i);


                    let ssbo_write_info = vk::WriteDescriptorSet::default()
                        .dst_set(sets[i])
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(std::slice::from_ref(&ssbo_info));

                    let img_infos = rgba_manager.descriptor_set_image_infos();

                    let other_img_infos_write_info = vk::WriteDescriptorSet::default()
                        .dst_set(sets[i])
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(img_infos.len() as _)
                        .image_info(&img_infos);

                    let td_buffer_info = material_manager.make_texture_descriptor_buffer(i);

                    let td_buffer_write_info = vk::WriteDescriptorSet::default()
                        .dst_set(sets[i])
                        .dst_binding(4)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(std::slice::from_ref(&td_buffer_info));

                    let glyph_buffer_info = ui_quad_buffers.buffer_descriptor(i);

                    let glyph_buffer_write_info = vk::WriteDescriptorSet::default()
                        .dst_set(sets[i])
                        .dst_binding(5)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(std::slice::from_ref(&glyph_buffer_info));

                    let write_infos = [buffer_write_info, image_write_info, ssbo_write_info, other_img_infos_write_info, td_buffer_write_info, glyph_buffer_write_info];

                    unsafe { vk_device.update_descriptor_sets(&write_infos, &[]) };
                }

                // setup attachments for dynamic render pass
                {
                    let color_attachment_info = vk::RenderingAttachmentInfo::default()
                        .image_view(sc_views[image_index as usize])
                        .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.45, 0.45, 0.45, 1.0],
                            },
                        });

                    let depth_attachment_info = vk::RenderingAttachmentInfo::default()
                        .image_view(*depth_img_view)
                        .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        });
                
                    let rendering_info = vk::RenderingInfo::default()
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: sc_extent,
                        })
                        .layer_count(1)
                        .color_attachments(std::slice::from_ref(&color_attachment_info))
                        .depth_attachment(&depth_attachment_info);

                    unsafe { vk_device.cmd_begin_rendering(cmd_buffer, &rendering_info) };
                }

                // 3D render geometry pipeline rendering
                {
                    // bind descriptor set
                    unsafe {vk_device.cmd_bind_descriptor_sets(cmd_buffer, 
                        vk::PipelineBindPoint::GRAPHICS, *m_g_mesh_prim_pipeline_layout, 
                        0, std::slice::from_ref(&apple_desc_sets[image_index as usize]), &[])}

                    // bind pipeline
                    unsafe { vk_device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, *g_mesh_prim_pipeline) };

                    // bind geometry buffers
                    {
                        let vertex_buffers = [*mesh_manager.buffer()];
                        let vb_offsets = [0 as u64];

                        unsafe { vk_device.cmd_bind_vertex_buffers(cmd_buffer, 0, &vertex_buffers, &vb_offsets)}
                        unsafe { vk_device.cmd_bind_index_buffer(cmd_buffer, *mesh_manager.buffer(), 0, vk::IndexType::UINT32) }
                    }

                    // setup indirection buffers
                    let indirect_slice = indirect_buffer.slice(image_index as usize);
                    let draw_count = render_prim_work.len() as u32;

                    for i in 0..draw_count {
                        let ri = &render_prim_work[i as usize];

                        indirect_slice[i as usize] = vk::DrawIndexedIndirectCommand {
                            index_count: ri.indices_view.get_element_count() as u32,
                            instance_count: 1,
                            first_index: ri.indices_view.buffer_index_offset() as u32,
                            vertex_offset: ri.vertices_view.buffer_index_offset() as i32,
                            first_instance: i as u32
                        };
                    }

                    // THE render draw command
                    let stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32;
                    unsafe { vk_device.cmd_draw_indexed_indirect(cmd_buffer, *indirect_buffer.buffer(image_index as usize), 0, draw_count, stride)};
                }


                // do font rendering
                for ui_quad_job in ui_quad_jobs {
                    let pc_job = UIQuadJobPushConstant {
                        color: ui_quad_job.color,
                        resolution: [sc_extent.width as f32, sc_extent.height as f32, 0.0, 0.0],
                        glyph_texture_index: rgba_manager.descriptor_index(q_font_img_id),
                        padding: [0 ; 3],
                    };
                    let pc_bytes = unsafe { std::slice::from_raw_parts(&pc_job as *const UIQuadJobPushConstant as *const u8, std::mem::size_of::<UIQuadJobPushConstant>()) };
                    match ui_quad_job.render_type {
                        UiQuadRenderType::Glyphs => {
                            unsafe {vk_device.cmd_bind_descriptor_sets(cmd_buffer, 
                                vk::PipelineBindPoint::GRAPHICS, *m_g_font_ui_pipeline_layout, 
                                0, std::slice::from_ref(&apple_desc_sets[image_index as usize]), &[])}
                            unsafe { vk_device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, *g_font_ui_pipeline) };
                            unsafe { vk_device.cmd_push_constants(cmd_buffer, *m_g_font_ui_pipeline_layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_bytes); }
                        },
                        UiQuadRenderType::SolidColor => {
                            unsafe {vk_device.cmd_bind_descriptor_sets(cmd_buffer, 
                                vk::PipelineBindPoint::GRAPHICS, *m_g_solid_color_ui_pipeline_layout, 
                                0, std::slice::from_ref(&apple_desc_sets[image_index as usize]), &[])}
                            unsafe { vk_device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, *g_solid_color_ui_pipeline) };
                            unsafe { vk_device.cmd_push_constants(cmd_buffer, *m_g_solid_color_ui_pipeline_layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_bytes); }
                        },
                    }
                    let verts_per_quad = 6;
                    unsafe { vk_device.cmd_draw(cmd_buffer, ui_quad_job.quad_count * verts_per_quad, 1, ui_quad_job.quad_start_index * verts_per_quad, 0)};
                }

                unsafe { vk_device.cmd_end_rendering(cmd_buffer) };
            }

            // memory barrier for swapchain present
            {
                let color_mem_barrier_two = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::empty())
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(sc_images[image_index as usize])
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        layer_count: 1,
                        level_count: 1,
                        ..Default::default()
                    });

                let mem_barriers_two = [color_mem_barrier_two];

                let dependency_info = vk::DependencyInfo::default()
                    .image_memory_barriers(&mem_barriers_two);

                unsafe { vk_device.cmd_pipeline_barrier2(cmd_buffer, &dependency_info) };
            }

            unsafe { vk_device.end_command_buffer(cmd_buffer)? };
        }

        let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(prim_command_buffers[image_index as usize]);

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(std::slice::from_ref(&wait_semaphore_submit_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_submit_info))
            .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info));

        unsafe {
            vk_device.queue_submit2(
                queue,
                std::slice::from_ref(&submit_info),
                fence.clone(),
            )?
        };

        let signal_semaphores = [render_finished_semaphore.clone()];
        let swapchains = [swapchain_khr.clone()];
        let images_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&images_indices);

        let present_result = unsafe {
            swapchain
                .queue_present(queue, &present_info)
        };
        match present_result {
            Ok(is_suboptimal) if is_suboptimal => {
                // println!("is suboptimal");
                continue;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // println!("out of date");
                continue;
            }
            Err(error) => panic!("Failed to present queue. Cause: {}", error),
            _ => {}
        }
    }

    // wait for all work to be done before exiting function
    unsafe { vk_device.device_wait_idle()? }

    return Ok(());
}