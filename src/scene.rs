use std::{any::Any, collections::{HashMap, HashSet}, str::FromStr};

use anyhow::{anyhow, bail};
use glam::{Mat4, Quat, Vec3, Vec4};
use serde_json::{ Value};

use crate::{vk_ash_renderer::RenderFrameDesc, MeshAssetID, PrefabSceneID, RenderMeshDesc, SkinAssetID};


#[derive(Debug)]
pub struct MeshInstanceNode {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
    mesh_asset_id: MeshAssetID,
    opt_skin_asset_id: Option<SkinAssetID>,
}

type SerdeObject = serde_json::Map<String, serde_json::Value>;

pub trait SerialFood where Self: Sized
{
    fn makey(serde_obj: serde_json::Value) -> anyhow::Result<Option<Self>>;
}

fn extract_float(s_val: &serde_json::Value) -> anyhow::Result<f32>
{
    let json_num = s_val.as_number()
        .ok_or(anyhow!("JSON value is not a number"))?;
    let num = json_num.as_f64()
        .ok_or(anyhow!("JSON number is not float"))?;
    return Ok(num as f32);
}

fn check_fields_exact(obj: &SerdeObject, allowed_set: &HashSet::<&str>) -> anyhow::Result<()>
{
    let obj_fields: HashSet<&str> = obj.keys()
        // anything with # is ignored
        .filter(|s| s.starts_with("#") != true)
        .map(|s| s.as_str()).collect();
    if obj_fields != *allowed_set {
        bail!("extraneous fields found")
    }
    return Ok(());
}

fn extract_quat(serde_obj: &SerdeObject) -> anyhow::Result<Quat>
{
    let allowed_set = HashSet::from(["w", "x", "y", "z"]);
    check_fields_exact(serde_obj, &allowed_set)?;

    let obj_fields: HashSet<&str> = serde_obj.keys().map(|s| s.as_str()).collect();
    if obj_fields != allowed_set {
        bail!("extraneous fields found")
    }

    let xyzw = Vec4::new(
        extract_float(serde_obj.get("x").unwrap())?,
        extract_float(serde_obj.get("y").unwrap())?,
        extract_float(serde_obj.get("z").unwrap())?,
        extract_float(serde_obj.get("w").unwrap())?,
    );

    const ERROR_TOLERANCE: f32 = f32::EPSILON * 4.0;

    let length_error = (xyzw.length_squared() - 1.0).abs();

    if length_error > ERROR_TOLERANCE {
        bail!("Unormalized quaternion found.")
    }

    return Ok(Quat::from_vec4(xyzw).normalize());
}

fn extract_vec3(serde_obj: &SerdeObject) -> anyhow::Result<Vec3>
{
    // validate fields
    let allowed_set = HashSet::from(["x", "y", "z"]);
    check_fields_exact(serde_obj, &allowed_set)?;

    let x = extract_float(serde_obj.get("x").unwrap())?;
    let y = extract_float(serde_obj.get("y").unwrap())?;
    let z = extract_float(serde_obj.get("z").unwrap())?;

    return Ok(Vec3 {x, y, z});
}

pub trait FunPrefabNode {
    fn node_type_id() -> &'static str where Self: Sized;
    fn extract(serde_obj: &SerdeObject) -> anyhow::Result<Self> where Self: Sized;
    fn opt_render_desc(&self) -> Option<RenderMeshDesc> { None }
    fn opt_scene_instance(&self) -> Option<&SceneInstanceNode> { None }
}
#[derive(Debug)]
pub struct PivotNode {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}
impl FunPrefabNode for PivotNode {
    fn node_type_id() -> &'static str {
        "PivotNode"
    }

    fn extract(serde_obj: &SerdeObject) -> anyhow::Result<Self> {
        let allowed_set = HashSet::from(["position", "rotation", "scale"]);
        check_fields_exact(serde_obj, &allowed_set)?;

        let position = extract_vec3(&serde_obj.get("position").unwrap().as_object().unwrap())?;
        let rotation = extract_quat(&serde_obj.get("rotation").unwrap().as_object().unwrap())?;
        let scale = extract_vec3(&serde_obj.get("scale").unwrap().as_object().unwrap())?;
        
        Ok(PivotNode {
            position,
            rotation,
            scale,
        })
    }
}

#[derive(Debug)]
pub struct SceneInstanceNode
{
    pub position: Vec3,
    pub resource_asset_id: String,
    pub gltf_scene_index: u32,
}

impl FunPrefabNode for SceneInstanceNode {
    fn node_type_id() -> &'static str {
        "SceneInstanceNode"
    }

    fn extract(serde_obj: &SerdeObject) -> anyhow::Result<Self>
    {
        let allowed_set = HashSet::from(["position", "resource_asset_id", "gltf_index"]);
        check_fields_exact(serde_obj, &allowed_set)?;

        let pos_obj = 
            &serde_obj.get("position").unwrap()
            .as_object().ok_or(anyhow!(""))?;

        let position = extract_vec3(pos_obj)?;

        let resource_asset_id = serde_obj.get("resource_asset_id").unwrap()
            .as_str().ok_or(anyhow!(""))?
            .to_string();

        let gltf_scene_index = serde_obj.get("gltf_index").unwrap()
            .as_number().ok_or(anyhow!(""))?
            .as_u64().ok_or(anyhow!(""))? as u32;

        Ok(
            SceneInstanceNode {
                position,
                resource_asset_id,
                gltf_scene_index,
            }
        )
    }

    fn opt_scene_instance(&self) -> Option<&SceneInstanceNode> {
        Some(&self)
    }

}

type ExtractFn = dyn Fn(&SerdeObject) -> anyhow::Result<Box<dyn FunPrefabNode>>;

pub struct FunScene {
    pub nodes: Vec<Box<dyn FunPrefabNode>>,
}

pub fn funny() -> anyhow::Result<FunScene>
{
    let file_thing = include_str!("../assets/prefab_scenes/composite_thing.json");
    let thingy = serde_json::Value::from_str(file_thing)?;

    let base = thingy.as_object().ok_or(anyhow!(""))?;
    let json_nodes_array = base
        .get("nodes").ok_or(anyhow!(""))?
        .as_array().ok_or(anyhow!(""))?;

    let mut nodes = Vec::<Box<dyn FunPrefabNode>>::new();

    // let extract_vtable = HashMap::<&'static str, Box<ExtractFn>>::new();

    // fn register<T: FunPrefabNode + 'static>() {
    //     let type_id = T::node_type_id();
    //     let johnny = |serde_obj: &SerdeObject| -> anyhow::Result<_> {
    //         let george: Box<dyn FunPrefabNode> = Box::new(T::extract(serde_obj)?);
    //         return Ok(george);
    //     };
    //     let func: Box<ExtractFn> = Box::new(johnny);
        
    // }

    // let type_registry = HashMap::<String, >::new();

    for json_node in json_nodes_array {
        let node_obj = json_node.as_object().ok_or(anyhow!(""))?;
        let node_type_str = node_obj
            .get("#node_type").ok_or(anyhow!(""))?
            .as_str().ok_or(anyhow!(""))?;

        let abe: Box<dyn FunPrefabNode> = match node_type_str {
            x if x == PivotNode::node_type_id() 
                => Box::new(PivotNode::extract(node_obj)?),
            x if x == SceneInstanceNode::node_type_id() 
                => Box::new(SceneInstanceNode::extract(node_obj)?),
            _ => panic!("fuck")
        };

        nodes.push(abe);
    }

    dbg!(nodes.len());


    // let vecy = extract_pivot_node(thingy.as_object().unwrap())?;
    // println!("{:?}", vecy);

    Ok(FunScene {
        nodes
    })
}