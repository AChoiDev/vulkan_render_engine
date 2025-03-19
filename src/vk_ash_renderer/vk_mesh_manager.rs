use std::{collections::HashMap, marker::PhantomData};


use ash::vk;

use crate::mesh_primitive::MeshPrimitive;

use super::vk_bump_manager::{BumpBufferRange, BumpManager};

#[derive(Clone, Copy, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct RenderVertex {
    pub pos: [f32; 4],
    pub tex_coords: [f32; 4],
    pub normal: [f32; 4],
    pub tangent: [f32; 4],
}

impl RenderVertex {
    pub fn create_binding_information() -> (
        vk::VertexInputBindingDescription,
        Vec<vk::VertexInputAttributeDescription>,
    ) {
        let vert_binding_desc = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<RenderVertex>() as _)
            .input_rate(vk::VertexInputRate::VERTEX);

        let vert_pos_attribute_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(std::mem::offset_of!(RenderVertex, pos) as _);
        let tex_attribute_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(std::mem::offset_of!(RenderVertex, tex_coords) as _);
        let normal_attribute_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(std::mem::offset_of!(RenderVertex, normal) as _);
        let tangent_attribute_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(std::mem::offset_of!(RenderVertex, tangent) as _);

        let vert_attr_descs = vec![
            vert_pos_attribute_desc,
            tex_attribute_desc,
            normal_attribute_desc,
            tangent_attribute_desc,
        ];
        (vert_binding_desc, vert_attr_descs)
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct BoneVertex {
    pub pos: [f32; 4],
    pub joint_indices: [u32; 4],
    pub joint_weights: [f32; 4],
}
#[derive(Clone, Copy)]
pub struct IndexedVerticesAllocation {
    pub vertex_allocation_id: GeometryAllocationID,
    pub index_allocation_id: GeometryAllocationID,
}
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct GeometryAllocationID(usize);

#[derive(Clone, Copy)]
pub struct GeometryAllocationView<T: Sized> {
    pub byte_size: vk::DeviceSize,
    pub buffer_byte_offset: vk::DeviceSize,
    p: PhantomData<T>,
}

impl<T: Sized> GeometryAllocationView<T> {
    pub fn get_element_count(&self) -> u64 {
        self.byte_size / std::mem::size_of::<T>() as u64
    }

    pub fn buffer_index_offset(&self) -> u64 {
        self.buffer_byte_offset / std::mem::size_of::<T>() as u64
    }

    pub fn get_sub_view(
        &self,
        element_offset: u64,
        element_size: u64,
    ) -> GeometryAllocationView<T> {
        let sub_byte_offset = std::mem::size_of::<T>() as u64 * element_offset as u64;
        let sub_byte_size = std::mem::size_of::<T>() as u64 * element_size as u64;
        // todo: have asserts here for bounds checking
        GeometryAllocationView::<T> {
            byte_size: sub_byte_size,
            buffer_byte_offset: self.buffer_byte_offset + sub_byte_offset,
            p: PhantomData,
        }
    }
}

pub struct MeshManager<'a> {
    allocations: HashMap<GeometryAllocationID, BumpBufferRange>,
    internal_manager: BumpManager<'a>,
    alloc_counter: usize,
}
pub struct MeshByteUploadJob {
    pub data: Vec<u8>,
    pub buffer_byte_offset: u64,
}
const MAX_BUFFER_SIZE: u64 = 64 * 1024 * 1024; // 64 MB
const ALIGNMENT: u64 = 1024;
impl<'a> MeshManager<'a> {
    pub fn new(
        allocator: &'a vk_mem::Allocator,
        vk_device: &'a ash::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let usage = vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let manager = BumpManager::new(MAX_BUFFER_SIZE, allocator, vk_device, usage, ALIGNMENT)?;

        Ok(Self {
            allocations: HashMap::new(),
            internal_manager: manager,
            alloc_counter: 0,
        })
    }

    pub fn buffer(&self) -> &vk::Buffer {
        self.internal_manager.buffer()
    }

    pub fn allocate<GeometryType: Sized>(
        &mut self,
        element_count: u64,
    ) -> Result<GeometryAllocationID, Box<dyn std::error::Error>> {
        let req_element_size = std::mem::size_of::<GeometryType>() as u64 * element_count;
        let buffer_range = self
            .internal_manager
            .allocate(req_element_size, std::mem::size_of::<GeometryType>() as u64)?;

        let alloc_id = GeometryAllocationID(self.alloc_counter);
        self.allocations.insert(alloc_id, buffer_range);
        self.alloc_counter += 1;

        return Ok(alloc_id);
    }

    pub fn allocate_primitive(&mut self, primitive: &MeshPrimitive) 
    -> Result<(IndexedVerticesAllocation, [MeshByteUploadJob ; 2]), Box<dyn std::error::Error>> {
        // construct render vertices
        let mut render_vertices = Vec::new();

        for pos in primitive.positions.iter() {
            render_vertices.push(RenderVertex {
                pos: [pos[0], pos[1], pos[2], 0.0f32],
                tex_coords: [0.0, 0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            });
        }
        for (i, normal) in primitive.normals.iter().enumerate() {
            let padded_normal = [normal[0], normal[1], normal[2], 0.0];
            render_vertices[i].normal = padded_normal;
        }
        if let Some(tangent_list) = &primitive.tangents {
            for (i, tangent) in tangent_list.iter().enumerate() {
                render_vertices[i].tangent = *tangent;
            }
        }

        if let Some(tex_coord_list) = primitive.tex_coord_lists.iter().next() {
            for (i, tex_coord) in tex_coord_list.iter().enumerate() {
                let padded_tex_coords = [tex_coord[0], tex_coord[1], 0.0, 0.0];
                render_vertices[i].tex_coords = padded_tex_coords;
            }
        }

        // allocate on gpu side
        let vertices_alloc_id = self.allocate::<RenderVertex>(render_vertices.len() as u64)?;
        let indices_alloc_id = self.allocate::<u32>(primitive.indices.len() as u64)?;
        

        // TODO: make this panicless
        let vertex_alloc = self.get_view::<RenderVertex>(vertices_alloc_id).unwrap();
        let index_alloc = self.get_view::<u32>(indices_alloc_id).unwrap();

        Ok((
            IndexedVerticesAllocation {
                vertex_allocation_id: vertices_alloc_id,
                index_allocation_id: indices_alloc_id,
            },
            // defered upload jobs
            [
                MeshByteUploadJob {
                    data: bytemuck::cast_slice(render_vertices.as_slice()).to_vec(),
                    buffer_byte_offset: vertex_alloc.buffer_byte_offset,
                },

                MeshByteUploadJob {
                    data: bytemuck::cast_slice(primitive.indices.as_slice()).to_vec(),
                    buffer_byte_offset: index_alloc.buffer_byte_offset,
                },
            ]
        ))
    }

    pub fn get_view<T: Sized>(
        &self,
        id: GeometryAllocationID,
    ) -> Option<GeometryAllocationView<T>> {
        let range = self.allocations.get(&id)?;
        Some(GeometryAllocationView::<T> {
            byte_size: range.byte_size,
            buffer_byte_offset: range.byte_offset,
            p: PhantomData,
        })
    }
}
