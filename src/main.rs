// mod vk_renderer;
mod gltf_importer;
mod mesh_primitive;
mod quad_font;
mod ui_tree;
mod vk_ash_renderer;
mod animation;
mod scene;
mod console;

use std::{
    any::Any, collections::{HashMap, HashSet, VecDeque}, path::Path, sync::{
        mpsc::{Receiver, Sender},
        Arc, Mutex
    }, thread::sleep
};

use animation::{AnimationBinding, MyPrefabAnimation};
use console::handle_console;
use glam::{Mat4, Quat, Vec3};
use gltf_importer::GLTFLoad;
use mesh_primitive::MeshPrimitive;
use scene::PivotNode;
use vk_ash_renderer::{RenderFrameDesc, RenderResource};
use winit::{
    event::{ElementState, Event, WindowEvent},
    event_loop::{EventLoop, EventLoopBuilder},
    window::WindowBuilder,
};

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub struct ResourceAssetID {
    internal_str: &'static str
}

impl ResourceAssetID {
    fn new(input: &'static str) -> Self {
        // TODO: validate input string
        Self {
            internal_str: input
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub struct GltfAssetID {
    resource_id: ResourceAssetID,
    // assume gltf source
    doc_index: u32,
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
struct MeshAssetID {
    gltf: GltfAssetID,
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
struct SkinAssetID {
    gltf: GltfAssetID,
}

pub struct MyMesh {
    self_id: MeshAssetID,
    name: Option<String>,
    pub primitives: Vec<MeshPrimitive>,
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub struct MaterialAssetID {
    pub resource_id: ResourceAssetID,
    pub gltf_document_index: u32,
}


pub struct MyMaterial {
    pub name: Option<String>,
    pub self_id: MaterialAssetID,
    pub images: Vec<RgbaPbrImage>,
    pub render_archetype: RenderArchetype,
}

pub enum RenderArchetype {
    PBR,
    SolidUnlit
}

pub struct RgbaPbrImage {
    pub data: Vec<u8>,
    pub resolution_width: u32,
    pub resolution_hegiht: u32,
    pub slots: Vec<(PBRTextureSlot, RgbaByteOffset)>
}
#[derive(Clone, Copy, Debug)]
pub enum RgbaByteOffset {
    N0,
    N1,
    N2,
    N3,
}

impl RgbaByteOffset {
    pub fn num(&self) -> usize {
        match self {
            RgbaByteOffset::N0 => 0,
            RgbaByteOffset::N1 => 1,
            RgbaByteOffset::N2 => 2,
            RgbaByteOffset::N3 => 3,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum PBRTextureSlot {
    BaseColor,
    Emissive,
    Normal,
    Tangent,
    Opacity,
    Metallic,
    Roughness,
    Occlusion,
    Cavity,
}

#[derive(PartialEq, Eq, Hash)]
enum RGBTextureAttributeSize {
    RGB,
    RG,
    SingleChannel,
}

impl PBRTextureSlot {
    fn size(&self) -> RGBTextureAttributeSize {
        use PBRTextureSlot::*;
        use RGBTextureAttributeSize::*;

        match &self {
            BaseColor => RGB,
            Emissive => RGB,
            Normal => RG,
            Tangent => RG,
            Metallic => SingleChannel,
            Roughness => SingleChannel,
            Cavity => SingleChannel,
            Opacity => SingleChannel,
            Occlusion => SingleChannel,
        }
    }
}

#[derive(Debug)]
enum WinitUserEvent {
    WorldEnded,
}

pub struct RenderWorkTasker {
    available_work: Mutex<Option<RenderFrameDesc>>,
}

impl RenderWorkTasker {
    pub fn new() -> Self {
        Self {
            available_work: Mutex::new(None),
        }
    }

    pub fn try_get_work(&self) -> Option<RenderFrameDesc> {
        let mut guard = self.available_work.lock().unwrap();
        guard.take()
    }

    pub fn set_work(&self, work: RenderFrameDesc) {
        let mut guard = self.available_work.lock().unwrap();
        *guard = Some(work);
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
enum MyKeyCode {
    ArrowLeft,
    ArrowRight,
    ArrowUp,
    ArrowDown,
    Backquote,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
enum KeyState {
    Pressed,
    Released,
}

#[derive(Debug)]
enum TextTypeInput {
    Graphic(char),
    Backspace,
    Enter,
    // Tab,
    // Space,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    ArrowUp,
}

impl TextTypeInput {
    fn new(key_event: &winit::event::KeyEvent) -> Option<Self> {
        if key_event.state == ElementState::Pressed {
            match &key_event.logical_key {
                winit::keyboard::Key::Named(named_key) => {
                    let opt_text_type_input = match named_key {
                        winit::keyboard::NamedKey::Enter => Some(TextTypeInput::Enter),
                        winit::keyboard::NamedKey::Tab => Some(TextTypeInput::Graphic('\t')),
                        winit::keyboard::NamedKey::Space => Some(TextTypeInput::Graphic(' ')),
                        winit::keyboard::NamedKey::ArrowDown => Some(TextTypeInput::ArrowDown),
                        winit::keyboard::NamedKey::ArrowLeft => Some(TextTypeInput::ArrowLeft),
                        winit::keyboard::NamedKey::ArrowRight => Some(TextTypeInput::ArrowRight),
                        winit::keyboard::NamedKey::ArrowUp => Some(TextTypeInput::ArrowUp),

                        // winit::keyboard::NamedKey::End => todo!(),
                        // winit::keyboard::NamedKey::Home => todo!(),

                        // winit::keyboard::NamedKey::PageDown => todo!(),
                        // winit::keyboard::NamedKey::PageUp => todo!(),
                        winit::keyboard::NamedKey::Backspace => Some(TextTypeInput::Backspace),
                        // winit::keyboard::NamedKey::Clear => todo!(),
                        // winit::keyboard::NamedKey::Copy => todo!(),
                        // winit::keyboard::NamedKey::Cut => todo!(),
                        // winit::keyboard::NamedKey::Delete => todo!(),
                        // winit::keyboard::NamedKey::Insert => todo!(),
                        // winit::keyboard::NamedKey::Paste => todo!(),
                        // winit::keyboard::NamedKey::Redo => todo!(),
                        // winit::keyboard::NamedKey::Undo => todo!(),
                        _ => None,
                    };

                    return opt_text_type_input;
                }
                winit::keyboard::Key::Character(key_txt) => {
                    if key_txt.len() == 1 {
                        let key_char = key_txt.chars().next().unwrap();
                        if key_char.is_ascii_graphic() {
                            return Some(TextTypeInput::Graphic(key_char));
                        }
                    }
                }
                _ => {}
            }
        }

        return None;
    }
}

impl MyKeyCode {
    fn try_from_winit_key_code(keycode: winit::keyboard::KeyCode) -> Option<MyKeyCode> {
        use winit::keyboard::KeyCode as KC;
        match keycode {
            KC::ArrowLeft => Some(Self::ArrowLeft),
            KC::ArrowRight => Some(Self::ArrowRight),
            KC::ArrowUp => Some(Self::ArrowUp),
            KC::ArrowDown => Some(Self::ArrowDown),
            KC::Backquote => Some(Self::Backquote),
            _ => None,
        }
    }
}

fn main() {

    // todo better event loop handling
    let event_loop: EventLoop<WinitUserEvent> = EventLoopBuilder::with_user_event()
        .build()
        .expect("Failed to create event loop");

    let el_proxy = event_loop.create_proxy();

    let (tx, rx) = std::sync::mpsc::channel::<winit::window::Window>();
    let (tx_render_shutdown, rx_render_shutdown) = std::sync::mpsc::channel();
    let (tx_world_window, rx_world_window) = std::sync::mpsc::channel();
    let (tx_world_node_list, rx_world_node_list) = std::sync::mpsc::channel();
    let (tx_load_req, rx_load_req) = std::sync::mpsc::channel();
    let (tx_render_re, rx_render_re) = std::sync::mpsc::channel();
    let (tx_input, rx_input) = std::sync::mpsc::channel();

    let render_frame_work: Arc<RenderWorkTasker> = Arc::new(RenderWorkTasker::new());
    let render_frame_work_two: Arc<RenderWorkTasker> = render_frame_work.clone();

    // let vk_req_inst_exts = vulkano::swapchain::Surface::required_extensions(&event_loop);

    let render_thr_handle = std::thread::spawn(move || {
        // match vk_renderer::run(vk_req_inst_exts, rx_render_shutdown, rx) {
        match vk_ash_renderer::run(rx_render_shutdown, rx, rx_render_re, render_frame_work) {
            Ok(_) => {
                println!("Vulkan renderer ended successfully");
            }
            Err(e) => {
                eprintln!("Vulkan renderer ended with failure: {e}");
            }
        }
    });

    let world_thr_handle = std::thread::spawn(move || {
        match manage_world(
            rx_world_window,
            rx_world_node_list,
            tx_load_req,
            rx_input,
            render_frame_work_two,
        ) {
            Ok(_) => {
                println!("World ended successfully");
            }
            Err(e) => {
                eprintln!("World ended with failure: {e}");
            }
        }
        el_proxy
            .send_event(WinitUserEvent::WorldEnded)
            .expect("Failed to send event to event loop");
    });

    let asset_thr_handle = std::thread::spawn(move || {
        match manage_assets(rx_load_req, tx_world_node_list, tx_render_re) {
            Ok(_) => {
                println!("Asset manager ended successfully");
            }
            Err(e) => {
                eprintln!("Asset ended with error: {e}");
            }
        }
    });

    let mut opt_thr_handles = Some(vec![render_thr_handle, world_thr_handle, asset_thr_handle]);

    // send a window to the renderer
    let initial_window = WindowBuilder::new()
        .with_position(winit::dpi::PhysicalPosition::new(0, 0))
        .with_inner_size(winit::dpi::LogicalSize::new(1600, 900))
        .build(&event_loop)
        .unwrap();
    tx.send(initial_window)
        .expect("Failed to send initial window to renderer");

    let el_run_result = event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                tx_world_window
                    .send(())
                    .expect("Failed to send window destroyed event to world");
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event, .. },
                ..
            } => {
                tx_input.send(event);
                // if repeat == false {
                //     if let Some(my_key_code) = MyKeyCode::try_from_winit_key_code(key_code) {
                //         let input_state = MyInputEvent::Play {
                //             code: my_key_code,
                //             state: match state {
                //                 winit::event::ElementState::Pressed => KeyState::Pressed,
                //                 winit::event::ElementState::Released => KeyState::Released,
                //             },
                //         };
                //         tx_input.send(input_state);
                //         // tx_input
                //         // println!("Received input event: {:?}, {:?}", my_input, state);
                //     }
                // }

                /*
                if state == ElementState::Pressed {
                    match logical_key {
                        winit::keyboard::Key::Named(named_key) => {
                            let opt_text_type_input = match named_key {
                                winit::keyboard::NamedKey::Enter => Some(TextTypeInput::Enter),
                                winit::keyboard::NamedKey::Tab => Some(TextTypeInput::Tab),
                                winit::keyboard::NamedKey::Space => Some(TextTypeInput::Space),
                                winit::keyboard::NamedKey::ArrowDown => {
                                    Some(TextTypeInput::ArrowDown)
                                }
                                winit::keyboard::NamedKey::ArrowLeft => {
                                    Some(TextTypeInput::ArrowLeft)
                                }
                                winit::keyboard::NamedKey::ArrowRight => {
                                    Some(TextTypeInput::ArrowRight)
                                }
                                winit::keyboard::NamedKey::ArrowUp => Some(TextTypeInput::ArrowUp),

                                // winit::keyboard::NamedKey::End => todo!(),
                                // winit::keyboard::NamedKey::Home => todo!(),

                                // winit::keyboard::NamedKey::PageDown => todo!(),
                                // winit::keyboard::NamedKey::PageUp => todo!(),
                                winit::keyboard::NamedKey::Backspace => {
                                    Some(TextTypeInput::Backspace)
                                }
                                // winit::keyboard::NamedKey::Clear => todo!(),
                                // winit::keyboard::NamedKey::Copy => todo!(),
                                // winit::keyboard::NamedKey::Cut => todo!(),
                                // winit::keyboard::NamedKey::Delete => todo!(),
                                // winit::keyboard::NamedKey::Insert => todo!(),
                                // winit::keyboard::NamedKey::Paste => todo!(),
                                // winit::keyboard::NamedKey::Redo => todo!(),
                                // winit::keyboard::NamedKey::Undo => todo!(),
                                _ => None,
                            };

                            if let Some(text_type_input) = opt_text_type_input {
                                tx_input.send(MyInputEvent::TextType(text_type_input));
                            }
                        }
                        winit::keyboard::Key::Character(key_txt) => {
                            if key_txt.len() == 1 {
                                let key_char = key_txt.chars().next().unwrap();
                                if key_char.is_ascii_graphic() {
                                    let graphic = TextTypeInput::Graphic(key_char);
                                    tx_input.send(MyInputEvent::TextType(graphic));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                */
            }
            Event::UserEvent(user_event) => match user_event {
                WinitUserEvent::WorldEnded => {
                    tx_render_shutdown
                        .send(())
                        .expect("Failed to send shutdown event to renderer");
                    let handles = opt_thr_handles.take().unwrap();
                    for handle in handles {
                        handle.join().expect("Failed to join thread");
                    }
                    target.exit();
                }
            },
            _ => (),
        }
    });

    if let Err(e) = el_run_result {
        eprintln!("Event loop ended with error: {e}");
    }
}

struct WorldAssetPacket {
    resource_id: ResourceAssetID,
    scenes: HashMap<PrefabSceneID, PrefabScene>,
    animations: HashMap<PrefabAnimationID, MyPrefabAnimation>,
}

fn manage_assets(
    load_request_receiver: Receiver<ResourceAssetID>,
    node_hierarchy_sender: Sender<WorldAssetPacket>,
    render_re_sender: Sender<RenderResource>,
) -> Result<(), Box<dyn std::error::Error>> {
    struct GLTFAssetDesc {
        gltf_path: String,
        scale: f32,
    }

    let mut resource_registry = HashMap::<ResourceAssetID, GLTFAssetDesc>::new();
    // resource_registry.insert(ResourceAssetID {base_id: 100}, AssetDesc {gltf_path: "assets/BoomBox.glb".into()});
    resource_registry.insert(
        ResourceAssetID::new("AntiqueCamera"),
        GLTFAssetDesc {
            gltf_path: "assets/AntiqueCamera.glb".into(),
            scale: 1.0,
        },
    );
    resource_registry.insert(
        ResourceAssetID::new("BoomBox"),
        GLTFAssetDesc {
            gltf_path: "assets/BoomBox.glb".into(),
            scale: 200.0,
        },
    );
    resource_registry.insert(
        ResourceAssetID::new("SimpleSkin"),
        GLTFAssetDesc {
            gltf_path: "assets/SimpleSkin.gltf".into(),
            scale: 3.0,
        },
    );
    resource_registry.insert(
        ResourceAssetID::new("BoxAnimated"),
        GLTFAssetDesc {
            gltf_path: "assets/BoxAnimated.glb".into(),
            scale: 3.0,
        },
    );

    resource_registry.insert(
        ResourceAssetID::new("Fox"),
        GLTFAssetDesc {
            gltf_path: "assets/Fox.glb".into(),
            // gltf_path: "assets/RiggedSimple.glb".into(),
            // scale: 0.0,
            scale: 1.0,
        },
    );

    // resource_registry.insert(
    //     ResourceAssetID::new("Duck"),
    //     GLTFAssetDesc {
    //         gltf_path: "assets/Duck.glb".into(),
    //         scale: 1.0,
    //     },
    // );
    resource_registry.insert(
        ResourceAssetID::new("DamagedHelmet"),
        GLTFAssetDesc {
            gltf_path: "assets/DamagedHelmet.glb".into(),
            scale: 3.0,
        },
    );

    resource_registry.insert(
        ResourceAssetID::new("CesiumMilkTruck"),
        GLTFAssetDesc {
            gltf_path: "assets/CesiumMilkTruck.glb".into(),
            scale: 1.0,
        },
    );

    // resource_registry.insert(ResourceAssetID {base_id: 100}, AssetDesc {gltf_path: "assets/Box.glb".into()});

    loop {
        match load_request_receiver.try_recv() {
            Ok(resource_id) => {
                let asset_desc = resource_registry
                    .get(&resource_id)
                    .expect("Failed to find asset desc");
                let path_string = &asset_desc.gltf_path;

                let GLTFLoad {
                    resource_id,
                    scenes,
                    animations,
                    meshes,
                    materials,
                } =
                    gltf_importer::my_load(resource_id, Path::new(path_string), asset_desc.scale)?;

                let packet = WorldAssetPacket {
                    resource_id,
                    scenes,
                    animations,
                };

                node_hierarchy_sender.send(packet)?;

                let materials = materials.into_values().collect();
                let meshes = meshes.into_values().collect();

                let render_resource = (resource_id, meshes, materials);
                render_re_sender.send(render_resource)?;
            }
            Err(err) => match err {
                std::sync::mpsc::TryRecvError::Empty => {}
                std::sync::mpsc::TryRecvError::Disconnected => {
                    println!("World became disconnected.");
                    break;
                }
            },
        }
    }

    return Ok(());
}

#[derive(Clone)]
pub struct EulerAngles {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl From<EulerAngles> for Quat {
    fn from(value: EulerAngles) -> Self {
        Quat::from_euler(glam::EulerRot::XYZ, value.x, value.y, value.z)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PrefabNodeID {
    resource_id: ResourceAssetID,
    gltf_scene_index: u32,
    gltf_node_index: u32,
}


#[derive(Debug, Clone)]
struct PrefabNode {
    self_id: PrefabNodeID,

    opt_name : Option<String>,
    local_transform: glam::Mat4,
    opt_mesh_asset_id: Option<MeshAssetID>,
    opt_skin_asset_id: Option<SkinAssetID>,
}

struct RuntimeSkinBinding
{
    render_node: RuntimeNodeID,
    joint_nodes: Vec<RuntimeNodeID>,
    opt_skin_asset_id: Option<SkinAssetID>,
}

struct PrefabSkinBinding
{
    render_node: PrefabNodeID,
    joint_nodes: Vec<PrefabNodeID>,
    opt_skin_asset_id: Option<SkinAssetID>,
}

// node can return local transform
// node can return render descriptor

struct GoodNode
{
    opt_name : Option<String>,
    local_transform: glam::Mat4,
    opt_mesh_asset_id: Option<MeshAssetID>,
}


#[derive(Debug, Clone)]
struct RuntimeNode {
    opt_parent_node_id: Option<RuntimeNodeID>,

    opt_name : Option<String>,
    local_transform: glam::Mat4,
    opt_mesh_asset_id: Option<MeshAssetID>,
    opt_skin: Option<RuntimeSkin>,
}

#[derive(Debug, Clone)]
struct RuntimeSkin {
    // list of nodes in scene hierarchy along with inverse bind matrix
    joints: Vec<RuntimeJoint>,
}

#[derive(Debug, Clone)]
struct PrefabJoint {
    node: PrefabNodeID,
    inverse_bind_matrix: glam::Mat4,
}
#[derive(Debug, Clone)]
struct RuntimeJoint {
    node: RuntimeNodeID,
    inverse_bind_matrix: glam::Mat4,
}

#[derive(Clone)]
pub struct RenderMeshDesc {
    model_transform: glam::Mat4,
    mesh_asset_id: MeshAssetID,
    opt_joint_transforms: Option<Vec<glam::Mat4>>,
}
// #[derive(Clone)]
// pub struct NewerRenderMeshDesc {
//     model_transform: glam::Mat4,
//     geometry: MeshAssetID,
//     materials: Vec<MaterialAssetID>,
//     opt_joint_transform: Option<Vec<glam::Mat4>>
// }

#[derive(Clone, Debug)]
struct PrefabSkin {
    joints: Vec<PrefabJoint>,
}

struct PrefabScene {
    nodes: Vec<(PrefabNode, Option<PrefabNodeID>)>,
    skins: HashMap<SkinAssetID, PrefabSkin>,
    // INVARIANT: all animations referenced by a scene prefab must be under the same resource
    animations: HashSet<PrefabAnimationID>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PrefabSceneID {
    resource_id: ResourceAssetID,
    gltf_scene_index: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PrefabAnimationID {
    resource_id: ResourceAssetID,
    gltf_animation_index: u32,
}

struct PrefabInstantiationRequest {
    scenes: HashSet<PrefabSceneID>,
    opt_parent: Option<RuntimeNodeID>,
    opt_position: Option<Vec3>,
}

type RuntimeNodeID = generational_arena::Index;
type RuntimeScene = generational_arena::Arena<RuntimeNode>;

fn id_display(node_id: RuntimeNodeID) -> String {
    let (index, gen) = node_id.into_raw_parts();
    format!("i{}g{}", index, gen)
}

struct FunRuntimeScene 
{
    nodes: HashMap<std::any::TypeId, Box<dyn std::any::Any>>
}

impl FunRuntimeScene
{
    // fn yes<T>(&self)
    // {
    //     self.nodes.get(std::any::TypeId::of::<T>()).unwrap().downcast_ref::<Vec<T>>()
    // }
}



fn manage_world(
    window_destroyed_recv: Receiver<()>,
    node_hierarchy_recv: Receiver<WorldAssetPacket>,
    load_request_sender: Sender<ResourceAssetID>,
    input_recv: Receiver<winit::event::KeyEvent>,
    render_frame_work: Arc<RenderWorkTasker>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("World started");

    let mut prefab_scene_list = HashMap::<PrefabSceneID, PrefabScene>::new();
    let mut prefab_animation_list = HashMap::<PrefabAnimationID, MyPrefabAnimation>::new();

    let mut runtime_scene = RuntimeScene::new();
    let mut runtime_animations = Vec::<AnimationBinding>::new();

    // let mut special_joint_node_id: Option<_> = None;

    let pivot_nodes = generational_arena::Arena::<PivotNode>::new();

    /*
    let instantiate_prefab_scene =
        |prefab: &PrefabScene,
         runtime_scene: &mut RuntimeScene,
         opt_scen_insert_parent_id: Option<RuntimeNodeID>|
        //  s_joint_node_id: &mut Option<generational_arena::Index>| 
         {

            let mut skins_to_instantiate: Vec<(SkinAssetID, RuntimeNodeID)> = Vec::new();
            let mut prefab_node_list_idx_to_runtime_id = Vec::new();

            // loop through each node in the prefab scene
            for prefab_node_idx in 0..prefab.nodes.len() {
                let (prefab_node, opt_prefab_parent_index) = prefab.nodes[prefab_node_idx].clone();

                // get runtime parent id. if none exist, use insert_parent_id
                let opt_rt_parent_id = opt_prefab_parent_index
                    .map(|i| prefab_node_list_idx_to_runtime_id[i])
                    .or(opt_scen_insert_parent_id);

                let rt_node = RuntimeNode {
                    opt_name: prefab_node.opt_name,
                    opt_parent_node_id: opt_rt_parent_id,
                    opt_mesh_asset_id: prefab_node.opt_mesh_asset_id,
                    opt_skin: None,
                    local_transform: prefab_node.model_transform,
                };

                // insert node into scene
                let arena_id = runtime_scene.insert(rt_node);
                prefab_node_list_idx_to_runtime_id.push(arena_id);

                // Defer skin creation
                if let Some(skin_asset_id) = prefab_node.opt_skin_asset_id {
                    skins_to_instantiate.push((skin_asset_id, arena_id))
                }
            }

            // Skin Instantiation
            for (skin_asset_id, rt_node_id) in skins_to_instantiate {
                let prefab_skin = prefab.skins.get(&skin_asset_id).unwrap();

                // Create runtime joints by mapping prefab joints to runtime joints
                let mut joints = Vec::new();
                for prefab_joint in prefab_skin.joints.iter() {
                    let joint_rt_node_id = prefab_node_list_idx_to_runtime_id[prefab_joint.node as usize];
                    joints.push(RuntimeJoint {
                        node: joint_rt_node_id,
                        inverse_bind_matrix: prefab_joint.inverse_bind_matrix,
                    });
                }

                // Add joints to runtime node that needs skin
                let rt_node = runtime_scene.get_mut(rt_node_id).unwrap();
                rt_node.opt_skin = Some(RuntimeSkin { joints });
            }
        };
        */

    // let super_root_id = runtime_scene.insert(RuntimeNode {
    //     opt_parent_node_id: None,
    //     opt_mesh_asset_id: None,
    //     opt_skin: None,
    //     local_transform: glam::Mat4::IDENTITY,
    // });
    let parsed_scene = scene::funny()?;

    let mut tick_num = 0;

    let loop_start_time = std::time::Instant::now();
    let mut time_of_time_start = loop_start_time;

    let view_ratio = 16.0 / 9.0;
    // load_request_sender.send(ResourceAssetID {base_id: 100}).expect("Failed to send load request");
    // load_request_sender.send(ResourceAssetID {base_id: 101}).expect("Failed to send load request");
    load_request_sender
        .send(ResourceAssetID::new("Fox"))
        .expect("Failed to send load request");
    load_request_sender
        .send(ResourceAssetID::new("BoomBox"))
        .expect("Failed to send load request");
    load_request_sender
        .send(ResourceAssetID::new("DamagedHelmet"))
        .expect("Failed to send load request");
    load_request_sender
        .send(ResourceAssetID::new("AntiqueCamera"))
        .expect("Failed to send load request");

    let mut key_input_pressed_state = HashMap::<MyKeyCode, KeyState>::new();

    let mut console_input = "".to_string();
    let mut console_back_lines = Vec::<String>::new();
    let mut console_open = false;


    loop {
        match window_destroyed_recv.try_recv() {
            Ok(_) => {
                println!("World received window destroyed event");
                break;
            }
            Err(err) => match err {
                std::sync::mpsc::TryRecvError::Empty => {}
                std::sync::mpsc::TryRecvError::Disconnected => {
                    println!("World became disconnected.");
                    break;
                }
            },
        }

        let mut instantiate_queue = VecDeque::<PrefabInstantiationRequest>::new();

        let t = time_of_time_start.elapsed().as_secs_f32();

        // input handling
        let input_events: Vec<_> = input_recv.try_iter().collect();

        let mut ui_nodes = Vec::new();

        // game logic
        {
            // console logic
            handle_console(input_events, &mut console_open, &mut console_input, &mut console_back_lines, &runtime_scene, &mut time_of_time_start, &mut ui_nodes);

            // animation logic
            {
                let time = t;
                let mut i = 0;
                for anim in runtime_animations.iter() {
                    if i == 2 {
                        anim.apply(&mut runtime_scene, time, &prefab_animation_list);
                    }
                    i += 1;
                }
            }

            if runtime_scene.len() == 0 {
                for node in parsed_scene.nodes.iter() {
                    if let Some(instance_node) = node.opt_scene_instance() {
                        // TODO parse
                        // TODO use RC instead of leak
                        let internal_str = instance_node.resource_asset_id.clone().leak();
                        let resource_id = ResourceAssetID { internal_str };
                        let prefab_id = PrefabSceneID { 
                            resource_id, 
                            gltf_scene_index: instance_node.gltf_scene_index
                        };
                        let mut scenes = HashSet::new();
                        scenes.insert(prefab_id);
                        let request = PrefabInstantiationRequest {
                            scenes,
                            opt_parent: None,
                            opt_position: Some(instance_node.position.clone()),
                        };
                        instantiate_queue.push_back(request);
                    } else {
                        
                    }
                }
            }
        }

        // obtain required resources
        let mut required_resources = HashSet::new();
        for request in instantiate_queue.iter() {
            for scene in request.scenes.iter() {
                required_resources.insert(scene.resource_id);
            }
        }

        // receive from asset server
        loop {
            node_hierarchy_recv
                .try_iter()
                .for_each(|packet| {
                    prefab_scene_list.extend(packet.scenes);
                    prefab_animation_list.extend(packet.animations);
                });

            let loaded_scene_resource_ids: HashSet<_> = prefab_scene_list.keys().map(|id| id.resource_id).collect();
            let loaded_animation_resource_ids: HashSet<_> = prefab_animation_list.keys().map(|id| id.resource_id).collect();
            let loaded_resources = loaded_scene_resource_ids.union(&loaded_animation_resource_ids).cloned().collect();

            if required_resources.is_subset(&loaded_resources) {
                break;
            }
        }

        // instantiate from requests
        while let Some(request) = instantiate_queue.pop_front() {
            let mut prefab_node_id_to_runtime_id = HashMap::<PrefabNodeID, RuntimeNodeID>::new();
            let mut animations = HashSet::new();

            // instantiate scenes
            for scene in request.scenes.iter() {
                let prefab = prefab_scene_list
                    .get(&scene)
                    .expect("Failed to find prefab");

                let mut skins_to_instantiate: Vec<(SkinAssetID, RuntimeNodeID)> = Vec::new();

                // loop through each node in the prefab scene
                for (prefab_node, opt_node_parent_index) in prefab.nodes.iter() {

                    // get runtime parent id. if none exist, use insert_parent_id
                    let opt_rt_parent_id = opt_node_parent_index.clone()
                        .map(|i| { 
                            prefab_node_id_to_runtime_id[&i]
                        })
                        .or(request.opt_parent);

                    let root_transform = Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, request.opt_position.unwrap_or(Vec3::ZERO));

                    let rt_node = RuntimeNode {
                        opt_name: prefab_node.opt_name.clone(),
                        opt_parent_node_id: opt_rt_parent_id,
                        opt_mesh_asset_id: prefab_node.opt_mesh_asset_id,
                        opt_skin: None,
                        local_transform: root_transform * prefab_node.local_transform,
                    };

                    // insert node into scene
                    let arena_id = runtime_scene.insert(rt_node);
                    prefab_node_id_to_runtime_id.insert(prefab_node.self_id.clone(), arena_id);
                    // prefab_node_list_idx_to_runtime_id.push(arena_id);

                    // Defer skin creation
                    if let Some(skin_asset_id) = prefab_node.opt_skin_asset_id {
                        skins_to_instantiate.push((skin_asset_id, arena_id))
                    }
                }

                // Skin Instantiation
                for (skin_asset_id, rt_node_id) in skins_to_instantiate {
                    let prefab_skin = prefab.skins.get(&skin_asset_id).unwrap();

                    // Create runtime joints by mapping prefab joints to runtime joints
                    let mut joints = Vec::new();
                    for prefab_joint in prefab_skin.joints.iter() {
                        let joint_rt_node_id = prefab_node_id_to_runtime_id[&prefab_joint.node];
                        joints.push(RuntimeJoint {
                            node: joint_rt_node_id,
                            inverse_bind_matrix: prefab_joint.inverse_bind_matrix,
                        });
                    }

                    // Add joints to runtime node that needs skin
                    let rt_node = runtime_scene.get_mut(rt_node_id).unwrap();
                    rt_node.opt_skin = Some(RuntimeSkin { joints });
                }

                animations.extend(prefab.animations.clone());
            }

            // instantiate animations
            for anim_id in animations.iter() {
                let prefab_animation = prefab_animation_list.get(anim_id).unwrap();
                let mut r_channels = Vec::new();
                for p_channel in prefab_animation.channels.iter() {
                    let target_node = prefab_node_id_to_runtime_id[&p_channel.target];
                    r_channels.push(target_node);
                }

                let r_animation = AnimationBinding {
                    channel_node_bindings: r_channels,
                    animation_id: anim_id.clone(),
                };
                runtime_animations.push(r_animation);
            }
        }

        let compute_global_transform = |node: &RuntimeNode| -> glam::Mat4 {
            let mut global_transform = node.local_transform;
            // traverse parents up hierarchy
            let mut opt_current_idx = node.opt_parent_node_id;
            while let Some(current_idx) = opt_current_idx {
                let curr_node = &runtime_scene[current_idx];
                global_transform = curr_node.local_transform * global_transform;
                opt_current_idx = runtime_scene[current_idx].opt_parent_node_id;
            }
            global_transform
        };

        // generate render frame work
        let mut render_mesh_decs = Vec::new();
        for (_, node) in runtime_scene.iter() {

            let mesh_asset_id = match node.opt_mesh_asset_id {
                Some(val) => val,
                None => continue, // skip iteration since this node doesnt render meshes
            };

            // extract the RenderMeshDesc
            let mut global_transform = compute_global_transform(&node);

            let opt_joint_transforms = if let Some(skin) = &node.opt_skin {
                global_transform = glam::Mat4::IDENTITY;
                // skin is present. calculate joint matrices
                let joint_matrices: Vec<_> = 
                    skin
                    .joints
                    .iter()
                    .map(|joint| {
                        // joint transform calculation
                        let joint_node = &runtime_scene[joint.node];
                        let joint_gl_transform = compute_global_transform(joint_node);

                        joint_gl_transform * joint.inverse_bind_matrix
                    })
                    .collect();

                Some(joint_matrices)
            } else {
                None
            };

            let mesh_desc = RenderMeshDesc {
                model_transform: global_transform,
                mesh_asset_id,
                opt_joint_transforms,
            };

            render_mesh_decs.push(mesh_desc);
        }

        // camera transforms
        let (view_transform, projection_transform) = {
            let source = glam::vec3(5.0, 8.0 + t.sin() * 1.0, 5.0) * 2.2;
            let dest = glam::vec3(0.0, source.y, 0.0);
            let look_dir = (dest - source).normalize();

            let look_up_dir = look_dir
                .cross(glam::vec3(0.0, 1.0, 0.0))
                .cross(look_dir)
                .normalize();

            let view = glam::Mat4::look_at_lh(source, look_dir, look_up_dir);
            let mut proj =
                glam::Mat4::perspective_lh(90.0f32.to_radians(), view_ratio, 0.05f32, 1000.0f32);
            proj.y_axis.y *= -1.0;

            (view, proj)
        };

        let frame_desc = RenderFrameDesc {
            tick_num,
            view_transform,
            projection_transform,
            mesh_instances: render_mesh_decs,
            ui_nodes,
        };

        render_frame_work.set_work(frame_desc);

        sleep(std::time::Duration::from_millis(25));

        tick_num += 1;
    }
    return Ok(());
}
