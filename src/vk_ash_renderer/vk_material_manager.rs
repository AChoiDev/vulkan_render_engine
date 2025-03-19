use std::collections::HashMap;

use ash::vk;

use crate::{vk_ash_renderer::vk_rgba_manager::PowerTwoTextureLength, MaterialAssetID, MyMaterial};

use super::{mortal::Mortal, vk_rgba_manager::{RgbaAllocID, RgbaManager}, vk_slab_manager::DynamicFlightedArray, FRAMES_IN_FLIGHT};

type TextureGroupDesc = [u32 ; 8];

pub struct MaterialManager<'a> {
    texture_descriptor_flighted_buffer: DynamicFlightedArray::<'a, TextureGroupDesc, FRAMES_IN_FLIGHT>,
    texture_groups: Vec<TextureGroupDesc>,
    material_id_to_texture_group_index: HashMap<MaterialAssetID, usize>,
}

impl<'a> MaterialManager<'a> {
    pub fn new(vma_allocator: &'a vk_mem::Allocator, vk_device: &'a ash::Device) -> Result<Self, Box<dyn std::error::Error>> {
        let dynamic_buffer = DynamicFlightedArray::<TextureGroupDesc, FRAMES_IN_FLIGHT>::new(
            300, vma_allocator, vk_device, 
            vk::BufferUsageFlags::STORAGE_BUFFER)?;

        Ok(Self {
            texture_groups: Vec::new(),
            material_id_to_texture_group_index: HashMap::new(),
            texture_descriptor_flighted_buffer: dynamic_buffer,
        })
    }

    pub fn texture_group_index(&self, material_id: &MaterialAssetID) -> Option<usize> {
        self.material_id_to_texture_group_index.get(material_id).copied()
    }

    pub fn make_texture_descriptor_buffer(&mut self, index: usize) -> vk::DescriptorBufferInfo {
        let slice = self.texture_descriptor_flighted_buffer.slice(index);

        for (i, tex_group) in self.texture_groups.iter().enumerate() {
            slice[i] = tex_group.clone();
        }

        self.texture_descriptor_flighted_buffer.buffer_descriptor(index)
    }

    // If successful, returns a list of upload jobs
    pub fn add_material(&mut self, material: MyMaterial, rgba_manager: &mut RgbaManager) 
    -> Result<Vec<(Vec<u8>, RgbaAllocID)>, Box<dyn std::error::Error>> {
        let mut upload_jobs = Vec::new();

        let material_id = material.self_id;
        println!("received material id: {:?}", material_id);
        let default_tex_index = 0;
        let mut tex_group_desc = [default_tex_index ; 8];
        
        // TODO: make texturegroupdesc optional
        for img in material.images.into_iter() {
            let pt_texture_length = match img.resolution_width {
                2048 => PowerTwoTextureLength::L2048,
                1024 => PowerTwoTextureLength::L1024,
                _ => todo!("not supported"),
            };

            let alloc_id = rgba_manager.allocate(pt_texture_length)?;
            upload_jobs.push((img.data, alloc_id));

            let desc_index = rgba_manager.descriptor_index(alloc_id);

            for (slot, byte_offset) in &img.slots {
                use crate::PBRTextureSlot as Slot;
                let mut desc_val = desc_index as u32;

                let offset = byte_offset.num() as u32;
                debug_assert!(offset < 4);
                desc_val = desc_val | (offset << (32 - 2));

                let slot_index = 
                    match slot {
                        Slot::BaseColor => 0,
                        Slot::Normal => 1,
                        Slot::Emissive => 2,
                        Slot::Metallic => 3,
                        Slot::Roughness => 4,
                        Slot::Occlusion => 5,
                        Slot::Tangent => todo!(),
                        Slot::Opacity => todo!(),
                        Slot::Cavity => todo!(),
                    };
                tex_group_desc[slot_index] = desc_val;
            }
        }
        self.texture_groups.push(tex_group_desc);
        self.material_id_to_texture_group_index.insert(material_id, self.texture_groups.len() - 1);



        Ok(upload_jobs)
    }
}
