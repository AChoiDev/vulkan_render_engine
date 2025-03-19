

use super::mortal::Mortal;

use ash::vk;
use vk_mem::Alloc;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct BumpAllocationID(usize);

#[derive(Clone, Debug)]
pub struct BumpBufferRange {
    pub byte_offset: u64,
    pub byte_size: u64,
}

pub struct BumpManager<'a> {
    _allocation: Mortal<'a, vk_mem::Allocation>,
    buffer: Mortal<'a, vk::Buffer>,
    bytes_used: u64,
    total_byte_size: u64,
    byte_alignment: u64,
}

pub trait ByteSize {
    fn round_up_align(&self, alignment: u64) -> u64;
}

impl ByteSize for u64 {
    fn round_up_align(&self, alignment: u64) -> u64 {
        if self % alignment == 0 {
           *self
        } else {
            self - (self % alignment) + alignment
        }
    }
}

impl<'a> BumpManager<'a> {
    pub fn new(req_byte_size: u64, allocator: &'a vk_mem::Allocator, vk_device: &'a ash::Device, buffer_usage: vk::BufferUsageFlags, alignment: u64) 
    -> Result<Self, Box<dyn std::error::Error>> {
        

        let byte_size = req_byte_size.round_up_align(alignment);

        let mesh_buffer_info = vk::BufferCreateInfo::default()
            .size(byte_size)
            .usage(buffer_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let (buff, mem) = unsafe{
            let (buff, mem) = allocator.create_buffer(&mesh_buffer_info, &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::Auto,
                ..Default::default()
            })?;

            (
                Mortal::new(buff, |b| vk_device.destroy_buffer(*b, None)),
                Mortal::new(mem, |m| allocator.free_memory(m))
            )
        };

        Ok(Self {
            _allocation: mem,
            buffer: buff,
            bytes_used: 0,
            byte_alignment: alignment,
            total_byte_size: byte_size,
        })

    }

    pub fn buffer(&self) -> &vk::Buffer {
        &self.buffer
    }

    pub fn allocate(&mut self, req_byte_size: u64, req_alignment: u64)
    -> Result<BumpBufferRange, Box<dyn std::error::Error>> {
        assert!(req_alignment <= self.byte_alignment, "Requested alignment is greater than buffer alignment");
        assert!(req_alignment != 0, "Requested alignment is zero");

        // find the offset into the buffer that aligns the content after the current bytes_used
        let content_start_offset = self.bytes_used.round_up_align(req_alignment);

        // the total memory size of the allocation
        // includes the padding to align the content and the padding to align to the buffer alignment
        let allocation_size = ((content_start_offset - self.bytes_used) + req_byte_size).round_up_align(self.byte_alignment);

        if self.bytes_used + allocation_size > self.total_byte_size {
            return Err("Manager: Buffer full".into());
        }

        let view = BumpBufferRange{
            byte_offset: content_start_offset,
            byte_size: req_byte_size,
        };
        println!("bytes used: {}, offset: {content_start_offset}, size: {req_byte_size}, req_byte_size: {req_byte_size}, req_alignment: {req_alignment}", self.bytes_used);

        self.bytes_used += allocation_size;

        return Ok(view);
    }

    // treats all memory previously allocated by this manager as free
    pub fn _reset(&mut self) {
        self.bytes_used = 0;
    }
}