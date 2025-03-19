use crate::MaterialAssetID;


#[derive(Clone)]
pub struct JointWeights<const N : usize> {
    pub indices : [u8 ; N],
    pub weights: [f32 ; N]
}

// todo: maybe make normals and tangents only 2 f32?
pub struct MeshPrimitive {
    pub indices: Vec<u32>,
    pub positions: Vec<[f32 ; 3]>,
    pub tex_coord_lists: Vec<Vec<[f32 ; 2]>>,
    pub tangents: Option<Vec<[f32 ; 4]>>,
    pub normals: Vec<[f32 ; 3]>,
    pub opt_joints : Option<Vec<JointWeights<4>>>,
    pub material_id: Option<MaterialAssetID>
}

impl MeshPrimitive {
    // panics if 0 texture coordinates present
    pub fn with_generated_tangents(self) -> Self {
        struct TangentVertexDesc {
            tangent: [f32 ; 4],
            base_vertex_index: u32,
        }

        // Define a proxy struct that implements the mikktspace::Geometry trait
        struct MikkProxyGeometry<'a> {
            indices: &'a [u32],
            positions: &'a [[f32 ; 3]],
            normals: &'a [[f32 ; 3]],
            uvs: &'a [[f32 ; 2]],
            output: &'a mut Vec<TangentVertexDesc>,
        }

        impl<'a> mikktspace::Geometry for MikkProxyGeometry<'a> {
            fn num_faces(&self) -> usize {
                self.indices.len() / 3
            }
        
            fn num_vertices_of_face(&self, _face: usize) -> usize {
                3
            }
        
            fn position(&self, face: usize, vert: usize) -> [f32; 3] {
                self.positions[self.indices[face * 3 + vert] as usize]
            }
        
            fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
                self.normals[self.indices[face * 3 + vert] as usize]
            }
        
            fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
                self.uvs[self.indices[face * 3 + vert] as usize]
            }

            fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
                self.output[face * 3 + vert] = TangentVertexDesc{ tangent, base_vertex_index: self.indices[face*3 + vert] };
            }
        }

        let tri_count = self.indices.len() / 3;

        let mut mikkt_generated_tangents = Vec::new();
        for _ in 0..(tri_count * 3) {
            let empty = TangentVertexDesc { tangent: [0.0, 0.0, 0.0, 0.0], base_vertex_index: 0};
            mikkt_generated_tangents.push(empty);
        }

        const TEXTURE_COORDINATE_LIST_INDEX: usize = 0;

        let mut proxy_geo = MikkProxyGeometry {
            indices: &self.indices,
            positions: &self.positions,
            normals: &self.normals,
            uvs: &self.tex_coord_lists[TEXTURE_COORDINATE_LIST_INDEX],
            output: &mut mikkt_generated_tangents,
        };

        mikktspace::generate_tangents(&mut proxy_geo);
        
        // use generated weld data to create new primitive
        let mut welded_vertex_desc_list = Vec::<TangentVertexDesc>::new();
        let mut welded_index_list = Vec::<u32>::new();

        const TOLERANCE: f32 = f32::EPSILON * 10.0;
        let is_close = |a: &[f32 ; 4], b: &[f32 ; 4]| {
            (a[0] - b[0]).abs() < TOLERANCE &&
            (a[1] - b[1]).abs() < TOLERANCE &&
            (a[2] - b[2]).abs() < TOLERANCE &&
            (a[3] - b[3]).abs() < TOLERANCE
        };

        for vertex_desc in mikkt_generated_tangents {
            // find a vertex that is similar enough to this one
            let vert_match = welded_vertex_desc_list.iter().enumerate().filter(|(_, w_desc)| {
                w_desc.base_vertex_index == vertex_desc.base_vertex_index 
                && is_close(&vertex_desc.tangent, &w_desc.tangent)
            }).next();

            if let Some((welded_vert_list_idx, _)) = vert_match {
                // found that we already added this vertex
                // reuse index in index list
                welded_index_list.push(welded_vert_list_idx as u32);
            } else {
                // found that this is a new vertex
                // add new vertex, and add its index to index list
                welded_vertex_desc_list.push(vertex_desc);
                let new_index = (welded_vertex_desc_list.len() - 1) as u32;
                welded_index_list.push(new_index);
            }
        }

        // form new primitive based on prior welding data
        let mut welded_positions = Vec::new();
        let mut welded_normals = Vec::new();
        let mut welded_uvs = Vec::new();
        for _ in 0..self.tex_coord_lists.len() {
            welded_uvs.push(Vec::new());
        }
        let mut welded_tangents = Vec::new();
        let mut opt_welded_joints = None;
        if self.opt_joints.is_some() {
            opt_welded_joints = Some(Vec::new());
        }

        for desc in welded_vertex_desc_list {
            let tangent = desc.tangent;
            let bv_idx = desc.base_vertex_index as usize;
            welded_positions.push(self.positions[bv_idx]);
            welded_normals.push(self.normals[bv_idx]);
            for list_idx in 0..self.tex_coord_lists.len() {
                welded_uvs[list_idx].push(self.tex_coord_lists[list_idx][bv_idx]);
            }
            welded_tangents.push(tangent);

            if let Some(welded_joints) = &mut opt_welded_joints {
                let jw = self.opt_joints.as_ref().unwrap()[bv_idx].clone();
                welded_joints.push(jw);
            }
        }

        MeshPrimitive {
            indices: welded_index_list,
            positions: welded_positions,
            tex_coord_lists: welded_uvs,
            tangents: Some(welded_tangents),
            normals: welded_normals,
            opt_joints: opt_welded_joints,
            material_id: self.material_id,
        }
    }
}

