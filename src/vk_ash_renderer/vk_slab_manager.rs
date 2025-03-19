

use ash::vk;
use vk_mem::Alloc;

use super::mortal::Mortal;

struct FlightObject<'a, T> {
    slice: &'a mut [T],
    buffer: Mortal<'a, vk::Buffer>,
    _allocation: Mortal<'a, vk_mem::Allocation>,
}

pub struct DynamicFlightedArray<'a, T, const F: usize> {
    flight_objects: Vec<FlightObject<'a, T>>,
    flight_object_mem_size: u64,
}

impl<'a, T, const F: usize> DynamicFlightedArray<'a, T, F> {
    pub fn new(item_count: u64, allocator: &'a vk_mem::Allocator, vk_device: &'a ash::Device, usage: vk::BufferUsageFlags) 
    -> Result<Self, Box<dyn std::error::Error>> {
        let mem_size = item_count * std::mem::size_of::<T>() as u64;

        let mut flight_obj_vec = Vec::new();
        for _ in 0..F {

            let buffer_info = vk::BufferCreateInfo::default()
                .size(mem_size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let (buff, mem) = unsafe{
                let (buff, mem) = allocator.create_buffer(&buffer_info, &vk_mem::AllocationCreateInfo {
                    flags: vk_mem::AllocationCreateFlags::MAPPED 
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                    usage: vk_mem::MemoryUsage::Auto,
                    ..Default::default()
                })?;


                (
                    Mortal::new(buff, |b| vk_device.destroy_buffer(*b, None)),
                    Mortal::new(mem, |m| allocator.free_memory(m))
                )
            };

            let ptr = allocator.get_allocation_info(&mem).mapped_data as *mut T;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, item_count as _) };
            // let (manager, ptr) = BumpManager::new_mapped(mem_size, allocator, vk_device, usage, true)?;
            // let slice = unsafe {std::slice::from_raw_parts_mut(ptr as *mut T, item_count as _)};
            flight_obj_vec.push(FlightObject {
                buffer: buff, _allocation: mem, slice
            });
        }

        Ok(Self {
            flight_objects: flight_obj_vec,
            flight_object_mem_size: mem_size,
        })
    }

    pub fn buffer_descriptor(&self, in_flight_index: usize) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: *self.flight_objects[in_flight_index].buffer,
            offset: 0,
            range: self.flight_object_mem_size,
        }
    }

    pub fn slice(&mut self, in_flight_index: usize) -> &mut [T] {
        &mut self.flight_objects[in_flight_index].slice
    }

    pub fn buffer(&self, in_flight_index: usize) -> &vk::Buffer {
        &self.flight_objects[in_flight_index].buffer
    }

}
