use std::collections::{HashMap, HashSet, VecDeque};
use crate::{animation::{MyAnimChannelOutputList, MyAnimInterpolation, MyPrefabAnimChannel, MyPrefabAnimation}, mesh_primitive::JointWeights, GltfAssetID, MaterialAssetID, MeshAssetID, MeshPrimitive, MyMaterial, MyMesh, PBRTextureSlot, PrefabAnimationID, PrefabJoint, PrefabNode, PrefabNodeID, PrefabScene, PrefabSceneID, PrefabSkin, RGBTextureAttributeSize, RgbaByteOffset, RgbaPbrImage, RenderArchetype, SkinAssetID};

use super::ResourceAssetID;

type BoxErr = Box<dyn std::error::Error>;


/* 
// todo: make this a test case
    let prox_verts = [
        [1., 0.3, 1.],
        [1., 0.6, 1.],
        [1., 0.6, 1.8],
        [1.9, 0.6, 1.8],
        [1.9, 0.6, 1.],
        [1.9, 0.3, 1.],
    ];

    let prox_uvs = [
        [0., 1.],
        [0.2, 1.],
        [1., 1.],
        [1., 0.],
        [0.2, 0.],
        [0., 0.],
    ];

    let rootish = 1.0 / (2.0f32.sqrt());

    let prox_normals = [
        [0.0, 0.0, -1.0],
        [0.0, rootish, -rootish],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, rootish, -rootish],
        [0.0, 0.0, -1.0],
    ];

    let prox_indices = [
        1, 2, 3,
        1, 3, 4,
        0, 1, 4,
        0, 4, 5
    ];

*/


fn extract_from_skin(
    doc_skin: &gltf::Skin,
    resource_id: ResourceAssetID,
    gltf_scene_index: u32,
    // prefab_node_index_lookup: &HashMap<u32, u32>, // maps the doc index to the prefab node index
    buffer_datas: &[gltf::buffer::Data]) 
-> Result<PrefabSkin, BoxErr> {
    let reader = doc_skin.reader(|buffer| Some(buffer_datas[buffer.index()].0.as_slice()));
    // by default, inverse bind matrices are identity
    let mut joint_specs: Vec<_> = 
        doc_skin.joints()
        .map(|j| PrefabJoint {
            node: PrefabNodeID {
                resource_id,
                gltf_scene_index,
                gltf_node_index: j.index() as u32,
            },
            inverse_bind_matrix: glam::Mat4::IDENTITY
        })
        .collect();
    if let Some(bind_mat_iter) = reader.read_inverse_bind_matrices() {
        // this iterator could have a count higher than the joint count (read the spec)
        let inv_bind_mats = bind_mat_iter.collect::<Vec<_>>();
        for i in 0..joint_specs.len() {
            joint_specs[i].inverse_bind_matrix = glam::Mat4::from_cols_array_2d(&inv_bind_mats[i]);
        }
    }

    let skin = PrefabSkin{joints: joint_specs};

    Ok(skin)
}


fn extract_from_mesh(
    mesh_id: GltfAssetID,
    doc_mesh: gltf::Mesh,
    buffer_datas: &[gltf::buffer::Data]) 
-> Result<MyMesh, BoxErr> {

    let mut primitives = Vec::new();

    for primitive in doc_mesh.primitives() {
        if primitive.mode() != gltf::mesh::Mode::Triangles {
            return Err("Unsupported geometry mode detected.".into());
        }
        let reader = primitive.reader(|buffer| Some(buffer_datas[buffer.index()].0.as_slice()));


        let read_positions : Vec<_> = reader.read_positions()
            .map(|p| p.collect()).ok_or("Positions unavailable in mesh.")?;

        let read_indices : Vec<u32> = {
            if let Some(r_indices) = reader.read_indices() {
                r_indices.into_u32().collect()
            } else {
                (0u32..(read_positions.len() as u32)).into_iter().collect()
            }
        };

        // read normals. if they aren't present, calculate them
        let normals : Vec<[f32 ; 3]> = {
            let opt_read_normals: Option<Vec<_>> = reader.read_normals().map(|n| n.collect());
            if let Some(read_normals) = opt_read_normals {
                read_normals
            } else {
                // calculate vertex normals via average
                let mut calc_normals = Vec::new();
                for _ in 0..read_positions.len() {
                    calc_normals.push(glam::vec3(0.0, 0.0, 0.0))
                }

                for tri_i in 0..(read_indices.len() / 3) {
                    let vert_i_a = read_indices[tri_i * 3 + 0] as usize;
                    let vert_i_b = read_indices[tri_i * 3 + 1] as usize;
                    let vert_i_c = read_indices[tri_i * 3 + 2] as usize;

                    let pos_a = glam::Vec3::from_array(read_positions[vert_i_a]);
                    let pos_b = glam::Vec3::from_array(read_positions[vert_i_b]);
                    let pos_c = glam::Vec3::from_array(read_positions[vert_i_c]);

                    let edge_m = pos_a - pos_b;
                    let edge_n = pos_c - pos_b;

                    let normal_contribution = edge_m.cross(edge_n);

                    calc_normals[vert_i_a] += normal_contribution;
                    calc_normals[vert_i_b] += normal_contribution;
                    calc_normals[vert_i_c] += normal_contribution;
                }

                calc_normals.into_iter()
                    .map(|n| n.normalize().to_array()).collect()
            }
        };

        // Extract texture coordinate information
        let mut read_tex_coord_lists = Vec::new();
        let mut tex_coord_list_idx = 0;
        while let Some(r_tex_coords) = reader.read_tex_coords(tex_coord_list_idx) {
            let tex_coords : Vec<_> = r_tex_coords.into_f32().collect();
            read_tex_coord_lists.push(tex_coords);
            tex_coord_list_idx += 1;
        }

        // Check if there is a joint set greater than supported
        {
            let semantic_is_joint_error = |sem: gltf::Semantic| {
                if let gltf::Semantic::Joints(set_idx) = sem {
                    set_idx > 0
                } else {
                    false
                }
            };
            if primitive.attributes().any(|attr| semantic_is_joint_error(attr.0)) {
                return Err("More than four joints per vertex unsupported.".into());
            }
        }

        // extract joint/bone vertex weighting
        let joints = {
            if let Some(r_joint_set) = reader.read_joints(0) {
                use gltf::mesh::util::ReadJoints;
                let joint_sets: Vec<_> = 
                    match r_joint_set {
                        ReadJoints::U16(joint_set_iter) => {
                            let mut sets = Vec::new();
                            // convert to u8 array, return error if any value is greater than 255
                            for joint_set in joint_set_iter {
                                let mut byte_joint_set = [0, 0, 0, 0];
                                if joint_set.iter().any(|j| *j > 255) {
                                    return Err("Bone index greater than 255 found. Note: 256 is the max number of bones per skin.".into());
                                } else {
                                    for i in 0..4 {
                                        byte_joint_set[i] = joint_set[i] as u8;
                                    }
                                }
                                sets.push(byte_joint_set);
                            }

                            sets
                        },
                        ReadJoints::U8(joint_set_iter) => {
                            joint_set_iter.collect()
                        },
                    };
                
                let weight_set_iter = reader.read_weights(0).unwrap().into_f32();

                assert_eq!(joint_sets.len(), weight_set_iter.len());
                let pairs: Vec<_> = joint_sets.into_iter().zip(weight_set_iter.into_iter())
                    .map(|(a, b)| JointWeights{indices: a, weights: b}).collect();

                Some(pairs)
            } else {
                None
            }
        };

        let material_id = {
            if let Some(idx) = primitive.material().index() {
                let resource_id = mesh_id.resource_id;
                let gen_asset_id = MaterialAssetID {resource_id, gltf_document_index: idx as _};
                Some(gen_asset_id)
            } else {
                None
            }
        };

        // read tangents
        let read_tangents = {
            if let Some(tangents_reader) = reader.read_tangents() {
                let tangents : Vec<_> = tangents_reader.collect();
                Some(tangents)
            } else {
                None
            }
        };

        let mut primitive = MeshPrimitive {
            indices: read_indices, positions: read_positions, tex_coord_lists: read_tex_coord_lists, normals, opt_joints: joints, material_id, tangents: read_tangents
        };

        // todo: make tangent generation configurable
        if primitive.tex_coord_lists.len() >= 1 && primitive.tangents.is_none() {
            primitive = primitive.with_generated_tangents()
        }

        primitives.push(primitive);
    }

    let mesh = MyMesh {
        self_id: MeshAssetID { gltf: mesh_id },
        name: doc_mesh.name().map(|n| n.to_string()),
        primitives
    };

    return Ok(mesh);
}



// TODO: unit test this
// each texture slot maps to a texture index and an optional RGB channel
// also returns the number of textures to make
fn make_packing_scheme(slots: HashSet<PBRTextureSlot>) -> (HashMap<PBRTextureSlot, (usize, RgbaByteOffset)>, usize) {
    let mut slot_to_tex_index = HashMap::new();
    let mut tex_count = 0;

    let mut three_channels_used = Vec::new();
    // insert the RGB attributes
    let where_rgb = slots.iter().filter(|slot| slot.size() == RGBTextureAttributeSize::RGB);
    for slot in where_rgb {
        slot_to_tex_index.insert(*slot, (tex_count, RgbaByteOffset::N0));
        three_channels_used.push(tex_count);
        tex_count += 1;
    }

    // insert the RG attributes
    let where_rg = slots.iter().filter(|slot| slot.size() == RGBTextureAttributeSize::RG);
    let mut two_channels_used = Vec::new();
    for slot in where_rg {
        slot_to_tex_index.insert(*slot, (tex_count, RgbaByteOffset::N0));
        two_channels_used.push(tex_count);
        tex_count += 1;
    }

    // insert the single channel attributes
    let mut one_channel_used = Vec::new();
    let where_single = slots.iter().filter(|slot| slot.size() == RGBTextureAttributeSize::SingleChannel);
    for slot in where_single {
        if let Some(tex_id) = three_channels_used.pop() {
            // insert into last channel slot
            slot_to_tex_index.insert(*slot, (tex_id, RgbaByteOffset::N3));
        } else if let Some(tex_id) = two_channels_used.pop() {
            // insert into third channel slot
            slot_to_tex_index.insert(*slot, (tex_id, RgbaByteOffset::N2));
            three_channels_used.push(tex_id);
        } else if let Some(tex_id) = one_channel_used.pop() {
            // insert into second channel slot
            slot_to_tex_index.insert(*slot, (tex_id, RgbaByteOffset::N1));
            two_channels_used.push(tex_id);
        } else {
            // create new image
            slot_to_tex_index.insert(*slot, (tex_count, RgbaByteOffset::N0));
            one_channel_used.push(tex_count);
            tex_count += 1;
        }
    }

    assert_eq!(slot_to_tex_index.len(), slots.len());

    for (a, (b, c)) in slot_to_tex_index.iter() {
        println!("{:?}, {}, offset: {:?}", a, b, c.num());
    }



    return (slot_to_tex_index, tex_count);
}

fn extract_from_material(id: MaterialAssetID, doc_mat: &gltf::Material, img_datas: &Vec<gltf::image::Data>) 
-> Result<MyMaterial, BoxErr> {

    // Find the image index associated with each texture slot
    let mut slot_to_img_idx = HashMap::new();
    {
        let pbr_m_r = doc_mat.pbr_metallic_roughness();

        if let Some(base_color) = pbr_m_r.base_color_texture() {
            let lookup = base_color.texture().source().index();
            slot_to_img_idx.insert(PBRTextureSlot::BaseColor, lookup);
        } 

        if let Some(m_r_tex) = pbr_m_r.metallic_roughness_texture() {
            let lookup = m_r_tex.texture().source().index();
            slot_to_img_idx.insert(PBRTextureSlot::Metallic, lookup);
            slot_to_img_idx.insert(PBRTextureSlot::Roughness, lookup);
        }

        if let Some(normal) = doc_mat.normal_texture() {
            let id = normal.texture().source().index();
            slot_to_img_idx.insert(PBRTextureSlot::Normal, id);
        }

        if let Some(occ) = doc_mat.occlusion_texture() {
            let id = occ.texture().source().index();
            slot_to_img_idx.insert(PBRTextureSlot::Occlusion, id);
        }
        if let Some(emissive) = doc_mat.emissive_texture() {
            let id = emissive.texture().source().index();
            slot_to_img_idx.insert(PBRTextureSlot::Emissive, id);
        }
    }

    // Find common image width
    // check they are all one image res
    let unique_image_indices: HashSet<usize> = slot_to_img_idx.values().cloned().collect();
    let mut opt_common_img_width: Option<u32> = None;
    for img_idx in unique_image_indices {
        let img = &img_datas[img_idx];
        assert!(img.width == img.height);
        if let Some(common_img_width) = opt_common_img_width {
            assert!(common_img_width == img.width)
        } else {
            opt_common_img_width = Some(img.width);
        }
    }

    let slots_found: HashSet<PBRTextureSlot> = slot_to_img_idx.keys().map(|k| *k).collect();

    let (maps, tex_count) = make_packing_scheme(slots_found);

    // initialize white RGB images
    let mut rgba_images: Vec<RgbaPbrImage> = {

        if tex_count == 0 {
            Vec::new()
        } else {

            let img_width = opt_common_img_width.unwrap();
            // temp is an iterator producing PIXEL_COUNT
            // number of RGB texels
            let temp = (0..((img_width * img_width) * 4)).map(|_| 255u8);

            (0..tex_count).map(|_| RgbaPbrImage {
                data: temp.clone().collect(),
                resolution_width: img_width,
                resolution_hegiht: img_width,
                slots: Vec::new(),
            }
            ).collect()
        }
    };

    for (slot, (rgb_img_idx, offset_in_pack_texel)) in maps {
        let img_idx = slot_to_img_idx[&slot];
        let img = &img_datas[img_idx];
        let rgba_dest = &mut rgba_images[rgb_img_idx];
        rgba_dest.slots.push((slot, offset_in_pack_texel));

        use gltf::image::Format as IFormat;

        if img.format != IFormat::R8G8B8A8
            && img.format != IFormat::R8G8B8
            && img.format != IFormat::R8G8
            && img.format != IFormat::R8 {
            panic!("Image format not supported.");
        }
        println!("format: {:?}", img.format);

        use PBRTextureSlot as TS;
        let offset_in_gltf_texel = match slot {
            TS::Opacity => 3,
            TS::Metallic => 2,
            TS::Roughness => 1,
            _ => 0,
        };

        use RGBTextureAttributeSize as AttrSize;
        let attr_size = match slot.size() {
            AttrSize::RGB => 3,
            AttrSize::RG => 2,
            AttrSize::SingleChannel => 1,
        };

        let texel_comp_size = match img.format {
            IFormat::R8G8B8A8 => 4,
            IFormat::R8G8B8 => 3,
            IFormat::R8G8 => 2,
            IFormat::R8 => 1,
            _ => panic!("Unsupported image format.")
        };
        let rgba_comp_num = 4;

        for i in 0..(img.width as usize * img.width as usize) {
            for j in 0..attr_size {
                rgba_dest.data[i*rgba_comp_num + j + offset_in_pack_texel.num()]
                    = img.pixels[i*texel_comp_size + j + offset_in_gltf_texel];
            }
        }

    }

    let self_id = id;

    let material = MyMaterial {
        self_id,
        name: doc_mat.name().map(|s| s.to_string()),
        images: rgba_images,
        render_archetype: RenderArchetype::PBR,
    };

    Ok(material)
}





pub struct GLTFLoad {
    pub resource_id: ResourceAssetID,
    pub animations: HashMap<PrefabAnimationID, MyPrefabAnimation>,
    pub scenes: HashMap<PrefabSceneID, PrefabScene>,
    pub meshes: HashMap<MeshAssetID, MyMesh>,
    pub materials: HashMap<MaterialAssetID, MyMaterial>,
}

// todo: remove ids from result
pub fn my_load(resource_id: ResourceAssetID, path: &std::path::Path, size_multiplier: f32) 
-> Result<GLTFLoad,BoxErr> {
    println!("opening gltf file");
    let (document, buffer_datas, image_datas) = gltf::import(path)?;
    println!("finished opening gltf file");

    let scene = document.scenes().next().unwrap();
    // BFS traverse nodes
    // let mut  : Vec<OurNode> = Vec::new();
    let mut nodes_to_visit : VecDeque<(Option<PrefabNodeID>, gltf::Node)> = scene.nodes().map(|n| (None, n)).collect();
    let mut mesh_ids = HashSet::<GltfAssetID>::new();
    let mut skin_ids = HashSet::<GltfAssetID>::new();

    let mut ret_node_list: Vec<(PrefabNode, Option<PrefabNodeID>)> = Vec::new();
    // let mut prefab_node_index_lookup = HashMap::new();

    let root_scale_mat = glam::Mat4::from_scale(glam::Vec3::splat(size_multiplier));
    while let Some((opt_parent_id, node)) = nodes_to_visit.pop_back() {

        let node_transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix());

        let model_transform =
            if opt_parent_id.is_none() {
                root_scale_mat * node_transform
            } else {
                node_transform
            };

        let opt_mesh_id = node.mesh().map(|m| {
            let id = GltfAssetID {resource_id, doc_index: m.index() as _};
            mesh_ids.insert(id);
            id
        });

        let opt_skin_id = node.skin().map(|s| {
            let id = GltfAssetID {resource_id, doc_index: s.index() as _};
            skin_ids.insert(id);
            id
        });

        let node_id = PrefabNodeID {
            resource_id,
            gltf_scene_index: 0,
            gltf_node_index: node.index() as u32,
        };

        let our_node = PrefabNode {
            self_id: node_id.clone(),
            opt_name : node.name().map(|s| s.to_string()),
            local_transform: model_transform,
            opt_mesh_asset_id: opt_mesh_id.map(|i| MeshAssetID {gltf: i}),
            opt_skin_asset_id: opt_skin_id.map(|i| SkinAssetID {gltf: i}),
        };

        ret_node_list.push((our_node, opt_parent_id));

        for child in node.children() {
            nodes_to_visit.push_front((Some(node_id.clone()), child));
        }
    }


    // Extract Skins
    let marked_skins = document.skins().filter(|skin| {
        let agltf_id = GltfAssetID {resource_id, doc_index: skin.index() as _};
        skin_ids.contains(&agltf_id)
    });
    let mut skin_registry = HashMap::new();
    for doc_skin in marked_skins {
        let skin = extract_from_skin(&doc_skin, resource_id, 0, &buffer_datas)?;
        let skin_asset_id = SkinAssetID {gltf: GltfAssetID {resource_id, doc_index: doc_skin.index() as _}};
        skin_registry.insert(skin_asset_id, skin);
    }

    

    // Extract Meshes
    let marked_meshes = document.meshes().filter(|mesh| {
        let agltf_id = GltfAssetID {resource_id, doc_index: mesh.index() as _};
        mesh_ids.contains(&agltf_id)
    });
    let mut mesh_registry = HashMap::new();
    let mut material_indices: HashSet<u32> = HashSet::new();
    for mesh in marked_meshes {
        // let asset_id = AssetID {namespace: None, name: "hi".into()};
        let mesh_id = GltfAssetID { resource_id, doc_index: mesh.index() as _ };
        let my_mesh = extract_from_mesh(mesh_id, mesh, &buffer_datas)?;
    
        // obtain materials to extract
        for primitive in my_mesh.primitives.iter() {
            if let Some(mat_id) = &primitive.material_id {
                material_indices.insert(mat_id.gltf_document_index);
            }
        }

        mesh_registry.insert(MeshAssetID { gltf: mesh_id }, my_mesh);
    }
    println!("{}", mesh_registry.len());
    println!("finished meshes");

    // Extract Materials
    let materials = document.materials().collect::<Vec<_>>();
    let mut material_registry = HashMap::new();
    for material_idx in material_indices {
        let material: &gltf::Material = &materials[material_idx as usize];
        let material_id = MaterialAssetID {resource_id, gltf_document_index: material_idx};
        let my_mat = extract_from_material(material_id, material, &image_datas)?;

        material_registry.insert(material_id, my_mat);
    }

    // Extract Animations
    let mut animation_registry = HashMap::new();
    for (anim_idx, animation) in document.animations().enumerate() {
        let mut channels = Vec::new();
        for channel in animation.channels() {
            let reader = channel.reader(|buffer| Some(buffer_datas[buffer.index()].0.as_slice()));

            let channel_output: MyAnimChannelOutputList =
                reader.read_outputs().map(|outputs| {
                    use gltf::animation::util::ReadOutputs as RO;
                    match outputs {
                        RO::Translations(translations) => {
                            MyAnimChannelOutputList::Translation(translations.map(|a| glam::Vec3::from_array(a)).collect())
                        },
                        RO::Rotations(rotations) => {
                            MyAnimChannelOutputList::Rotation(rotations.into_f32().map(|a| glam::Quat::from_array(a)).collect())
                        },
                        RO::Scales(scales) => {
                            MyAnimChannelOutputList::Scale(scales.map(|a| glam::Vec3::from_array(a)).collect())
                        },
                        RO::MorphTargetWeights(mt_weights) => {
                            MyAnimChannelOutputList::MorphTargetWeight(mt_weights.into_f32().collect())
                        },
                    }
                    
                }).unwrap();

            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Linear => MyAnimInterpolation::Linear,
                gltf::animation::Interpolation::Step => MyAnimInterpolation::Step,
                gltf::animation::Interpolation::CubicSpline => MyAnimInterpolation::CubicSpline,
            };

            let target = PrefabNodeID {
                resource_id,
                gltf_scene_index: 0 as u32, // TODO: change this
                gltf_node_index: channel.target().node().index() as u32,
            };

            let channel = MyPrefabAnimChannel {
                interpolation,
                target,
                inputs: reader.read_inputs().unwrap().collect(),
                outputs: channel_output,
            };

            channels.push(channel);
        }

        let anim = MyPrefabAnimation {
            name: animation.name().map(|s| s.to_string()),
            channels
        };
        animation_registry.insert(PrefabAnimationID {resource_id, gltf_animation_index: anim_idx as u32}, anim);
    }

    // TODO: only include animations that reference this scene
    let scene_prefab = PrefabScene {
        nodes: ret_node_list,
        skins: skin_registry,
        animations: animation_registry.keys().cloned().collect()
    };

    let mut scene_registry = HashMap::new();
    let scene_id = PrefabSceneID {resource_id, gltf_scene_index: 0};
    scene_registry.insert(scene_id, scene_prefab);

    let load = GLTFLoad {
        resource_id,
        animations: animation_registry,
        scenes: scene_registry,
        meshes: mesh_registry,
        materials: material_registry,
    };

    return Ok(load);
}