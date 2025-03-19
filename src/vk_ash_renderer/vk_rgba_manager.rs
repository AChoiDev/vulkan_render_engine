
use ash::vk;
use vk_mem::Alloc;
use crate::vk_ash_renderer::mortal::Mortal;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct RgbaAllocID(u32);
pub struct RgbaImgView<'a> {
    vk_view: Mortal<'a, vk::ImageView>,
    image_index: u32,
    layer_index: u32,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum PowerTwoTextureLength {
    L1,
    L2,
    L4,
    L8,
    L16,
    L32,
    L64,
    L128,
    L256,
    L512,
    L1024,
    L2048,
    L4096,
}

impl PowerTwoTextureLength {
    fn to_num(&self) -> u32 {
        match self {
            Self::L1 => 1,
            Self::L2 => 2,
            Self::L4 => 4,
            Self::L8 => 8,
            Self::L16 => 16,
            Self::L32 => 32,
            Self::L64 => 64,
            Self::L128 => 128,
            Self::L256 => 256,
            Self::L512 => 512,
            Self::L1024 => 1024,
            Self::L2048 => 2048,
            Self::L4096 => 4096,
        }
    }

}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
enum Availability {
    Available,
    Unavailable,
}

// TODO: decouple images from allocations (this is good enough for now)
struct RGBALayeredImage<'a> {
    allocation: Mortal<'a, vk_mem::Allocation>,
    vk_image: Mortal<'a, vk::Image>,
    length: PowerTwoTextureLength,
    available_layers: [Availability ; TEXTURE_ARRAY_SIZE as usize],
}

pub struct RgbaManager<'a> {
    allocator: &'a vk_mem::Allocator,
    layererd_images: Vec<RGBALayeredImage<'a>>,
    vk_device: &'a ash::Device,
    image_views: [Option<RgbaImgView<'a>> ; MAX_TEXTURE_COUNT as usize]
}

pub const MAX_TEXTURE_COUNT: u32 = 256;
const TEXTURE_ARRAY_SIZE: u32 = 8;

impl<'a> RgbaManager<'a> {

    const ARRAY_NONE_REPEAT_VALUE: Option<RgbaImgView<'a>> = None;

    // returns the vk::Image and layer index of the image view
    pub fn vk_image_information(&self, id: RgbaAllocID) -> Option<(vk::Image, u32)> {
        self.image_views[id.0 as usize].as_ref().map(|view| (self.layererd_images[view.image_index as usize].vk_image.clone(), view.layer_index) )
    }

    pub fn buffer_to_image_info(&self, id: RgbaAllocID) -> vk::BufferImageCopy {
        let view = self.image_views[id.0 as usize].as_ref().unwrap();
        let img_length = self.layererd_images[view.image_index as usize].length.to_num();
        let img_layer = view.layer_index;

        let img_extent = vk::Extent3D {
                width: img_length,
                height: img_length,
                depth: 1,
            };

        vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .base_array_layer(img_layer)
                    .mip_level(0)
            )
            .image_extent(img_extent)
    }


    pub fn descriptor_pool_size(&self, in_flight_count: u32) 
    -> vk::DescriptorPoolSize {
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(in_flight_count as u32 * MAX_TEXTURE_COUNT)
    }

    // TODO: make it clear the view returned is temporary
    pub fn descriptor_set_image_infos(&self) 
    -> Vec<vk::DescriptorImageInfo> {
        let default_view = self.image_views[0].as_ref().unwrap();
        self.image_views.iter().map(|opt_view| {
            let view = opt_view.as_ref().map(|v| v.vk_view.clone()).unwrap_or(default_view.vk_view.clone());
            
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
        }).collect()
    }

    pub fn descriptor_index(&self, id: RgbaAllocID) -> u32 {
        id.0
    }

    pub fn new(allocator: &'a vk_mem::Allocator, vk_device: &'a ash::Device) 
    -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            allocator,
            layererd_images: Vec::new(),
            vk_device,
            image_views: [Self::ARRAY_NONE_REPEAT_VALUE; MAX_TEXTURE_COUNT as usize],
        })
    }

    pub fn allocate(&mut self, requested_image_length: PowerTwoTextureLength)
    -> Result<RgbaAllocID, Box<dyn std::error::Error>> {
        // Find image view slot that is none (not in use)
        let (image_view_idx, _) = 
            self.image_views.iter_mut().enumerate()
            .find(|(_, value)| value.is_none())
            .ok_or("No more image view slots available.")?;

        let (layered_img_index, layer_index) = {
            // Find layered images that have the requested length and have an available layer
            let mut valid_layered_images_iter = 
                self.layererd_images.iter().enumerate()
                .filter(|(_, img)| img.length == requested_image_length && img.available_layers.iter().any(|v| *v == Availability::Available));

            if let Some(first_layered_img) = valid_layered_images_iter.next() {
                // Case: at least one valid layered image found

                let (selected_img_idx, selected_img) = {
                    // find the layered image with the least available layers
                    let mut min_img = first_layered_img;
                    let count_available = |v: &RGBALayeredImage| 
                        v.available_layers.iter().filter(|v| **v == Availability::Available).count();

                    for other_img in valid_layered_images_iter {
                        let current_min_count = count_available(min_img.1);
                        let other_min_count = count_available(other_img.1);

                        if other_min_count < current_min_count {
                            min_img = other_img;
                        }
                    }
                    min_img
                };

                // use first available layer
                let layer_index = selected_img.available_layers.iter().position(|v| *v == Availability::Available).unwrap();
                (selected_img_idx, layer_index as u32)
            } else {
                // Case: no valid layered images available, make a new layered image
                let img_length = requested_image_length.to_num();

                let img_extent = vk::Extent3D {
                        width: img_length,
                        height: img_length,
                        depth: 1,
                    };

                let img_create_info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(img_extent)
                    .mip_levels(1)
                    .array_layers(TEXTURE_ARRAY_SIZE)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .samples(vk::SampleCountFlags::TYPE_1);

                let (allocation, image) = unsafe {
                    let (img, mem) = self.allocator.create_image(&img_create_info, &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::AutoPreferDevice,
                        ..Default::default()
                    })?;

                    (
                        Mortal::new(mem, |m| self.allocator.free_memory(m)),
                        Mortal::new(img, |i| self.vk_device.destroy_image(*i, None)),
                    )
                };

                let available_layers = [Availability::Available; TEXTURE_ARRAY_SIZE as usize];

                let new_layered_img = RGBALayeredImage {
                    allocation,
                    vk_image: image,
                    length: requested_image_length,
                    available_layers,
                };

                self.layererd_images.push(new_layered_img);
                (self.layererd_images.len() - 1, 0u32)
            }
        };

        let layered_img = &mut self.layererd_images[layered_img_index];
        
        // create new image view and insert into image view array
        let img_view_create_info = vk::ImageViewCreateInfo::default()
            .image(*layered_img.vk_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                layer_count: 1,
                level_count: 1,
                base_array_layer: layer_index,
                ..Default::default()
            });

        let img_view = Mortal::new(
            unsafe { self.vk_device.create_image_view(&img_view_create_info, None)? },
            |iv| unsafe { self.vk_device.destroy_image_view(*iv, None) }
        );

        let view = RgbaImgView {
            vk_view: img_view,
            image_index: layered_img_index as u32,
            layer_index,
        };

        layered_img.available_layers[layer_index as usize] = Availability::Unavailable;

        self.image_views[image_view_idx as usize] = Some(view);
        Ok(RgbaAllocID(image_view_idx as u32))
    }
        
}