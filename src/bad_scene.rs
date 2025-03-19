use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail};
use glam::{Quat, Vec3};
use pest::Parser;
use pest_derive::Parser;

use crate::{EulerAngles, MeshAssetID, PrefabSceneID, ResourceAssetID, SkinAssetID};
#[derive(Parser)]
#[grammar="scn_grammar.pest"]
struct SceneParser { }

pub struct ParsedScene {
    pub nodes: Vec<ParsedNode>
}

#[derive(Debug)]
pub enum ParsedNode {
    Add {
        node_type: String, 
        opt_name: Option<String>, 
        opt_parent_index: Option<usize>, 
    },
    Instance { 
        parent: Option<usize>, 
        prefab_id: PrefabSceneID,
        opt_position: Option<Vec3>,
    }
}

pub struct AppleScene {

}

pub trait EntityNode {
}

struct PivotNode {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}
struct MeshInstanceNode {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
    mesh_asset_id: MeshAssetID,
    opt_skin_asset_id: Option<SkinAssetID>,
}

pub fn parse_scene() -> anyhow::Result<ParsedScene> {
    let scene_txt = include_str!("../assets/prefab_scenes/bench.scn");
    let main = SceneParser::parse(Rule::main, scene_txt)?;

    let mut node_list = Vec::<ParsedNode>::new();
    for node in main {
        let node_rule = node.as_rule();
        let mut inner_rules = node.into_inner();
        match node_rule {
            Rule::add_node => {
                let add_node_type = 
                    inner_rules.next().unwrap() // unwrap to add node decl
                    .into_inner().next().unwrap() // unwrap to decl_name
                    .as_str().to_string();

                let header_field_map = extract_field_map(inner_rules.next().unwrap())?;
                if header_field_map.len() > 2 {
                    bail!("Too many parameters in add node header");
                }
                let opt_header_name =
                    if let Some(header_name_val) = header_field_map.get("name") {
                        match header_name_val {
                            FieldValue::String(str) => Some(str.to_string()),
                            _ => panic!(),
                        }
                    } else {
                        None
                    };

                let opt_parent_index =
                    if let Some(val) = header_field_map.get("child_of") {
                        match val {
                            FieldValue::Int(int) => Some(*int as usize),
                            _ => panic!(),
                        }
                    } else {
                        None
                    };

                let object_field_map = extract_field_map(inner_rules.next().unwrap())?;

                node_list.push(
                    ParsedNode::Add {
                        node_type: add_node_type,
                        opt_name: opt_header_name,
                        opt_parent_index,
                    }
                );


            },
            Rule::instance_node => {

                // parse scene ID
                let raw_scn_prefab_str = 
                    inner_rules.next().unwrap() // unwrap to instance decl
                    .into_inner().next().unwrap() // unwrap to string
                    .into_inner().next().unwrap() // unwrap to inner str
                    .as_str();
                
                let scn_str_elements = raw_scn_prefab_str.split(".").into_iter().collect::<Vec<_>>();
                if scn_str_elements.len() != 2 {
                    bail!("Invalid scene ID");
                }

                let scene_id = PrefabSceneID {
                    resource_id: ResourceAssetID::new(scn_str_elements[0]),
                    gltf_scene_index: scn_str_elements[1].trim_start_matches("scene_").parse::<u32>()?
                };

                let mut field_map = extract_field_map(inner_rules.next().unwrap())?;

                let opt_parent_index =
                    if let Some(val) = field_map.get("child_of") {
                        match val {
                            FieldValue::Int(int) => Some(*int as usize),
                            _ => panic!(),
                        }
                    } else {
                        None
                    };

                // TODO: validate parent index
                // TODO: validate prefab id

                let opt_position = 
                    if let Some(val) = field_map.remove("position") {
                        Some(Vec3::extract(val)?)
                    } else {
                        None
                    };

                node_list.push(
                    ParsedNode::Instance{
                        parent: opt_parent_index,
                        prefab_id: scene_id,
                        opt_position,
                    }
                );
            },
            Rule::EOI => { }
            _ => unreachable!()
        }
        
    }

    Ok(ParsedScene {
        nodes: node_list,
    })
}

#[derive(Debug)]
enum FieldValue<'a> {
    String(String),
    Int(i32),
    Float(f32),
    Boolean(bool),
    UnparsedObject{field_list: pest::iterators::Pair<'a, Rule>, opt_type: Option<String>}
}

type FieldMap<'a> = HashMap<String, FieldValue<'a>>;

// TODO: non recursive field extraction
trait FieldExtractable: Sized {
    fn extract<'a>(field_value: FieldValue<'a>) -> anyhow::Result<Self>;
}

impl FieldExtractable for f32 {
    fn extract<'a>(field_value: FieldValue<'a>) -> anyhow::Result<Self> {
        if let FieldValue::Float(val) = field_value {
            return Ok(val);
        } else {
            bail!("Could not extract f32 from field value")
        }
    }
}

trait MaybeInto<T> where Self: Sized {
    fn maybe_into(self) -> Option<T> {
        None
    }
}

impl<T: From<H>, H> MaybeInto<T> for H {
    fn maybe_into(self) -> Option<T> {
        Some(self.into())
    }
}

struct SerialTypeRegistry {
    
}

impl SerialTypeRegistry {
    fn new() {

    }

    fn convert() {

    }
}

impl FieldExtractable for Vec3 {
    fn extract<'a>(field_value: FieldValue<'a>) -> anyhow::Result<Self> {
        if let FieldValue::UnparsedObject {field_list, opt_type} = field_value {
            // check if every field that requires an extraction is also marked as extract
            // field needs extraction if it isnt marked as ignore and doesn't have a default value

            if opt_type.is_some_and(|str| str != "Vec3") {
                bail!("Type mismatch for Vec3");
            } else {
                let mut field_map = extract_field_map(field_list)?;
                let field_names = field_map.keys()
                    .map(|s| s.as_str())
                    .collect::<HashSet<&str>>();
                let type_field_set = HashSet::from(["x", "y", "z"]);
                if field_names.is_subset(&type_field_set) {
                    let x = 
                        if let Some(field_val) = field_map.remove("x") {
                            f32::extract(field_val)?
                        } else {
                            0.0
                        };
                    let y = 
                        if let Some(field_val) = field_map.remove("y") {
                            f32::extract(field_val)?
                        } else {
                            0.0
                        };
                    let z = 
                        if let Some(field_val) = field_map.remove("z") {
                            f32::extract(field_val)?
                        } else {
                            0.0
                        };
                    return Ok(Vec3 {
                        x,
                        y,
                        z,
                    });
                } else {
                    bail!("Fields not compatible with type Vec3");
                }
            }
        } else {
            bail!("Cannot convert value literal to Vec3");
        }
    }
}


// trait Apple {

// }
// impl<A: Into<B>> Apple for A {

// }

// fn extract_general<'a, T: Sized>(field_value: FieldValue<'a>, value_type_name: &str) -> anyhow::Result<T> {
//     Ok(match value_type_name {
//         "Vec3" => Vec3::extract(field_value)?.into()?,
//     })
// }

fn extract_field_map(field_list_rule: pest::iterators::Pair<'_, Rule>) -> anyhow::Result<FieldMap> {
    assert!(field_list_rule.as_rule() == Rule::field_list);

    let mut field_map = HashMap::<String, FieldValue>::new();
    for field in field_list_rule.into_inner() {
        let mut field_parts = field.into_inner();
        let mut field_decl_parts = field_parts.next().unwrap().into_inner();
        let field_name = field_decl_parts.next().unwrap();
        let field_type = field_decl_parts.next().map(|name| name.as_str().to_string());
        // TODO: check if field type matches literal values?
    
        // map pest rule to field value enum
        let field_value_raw = field_parts.next().unwrap();
        let field_value = match field_value_raw.as_rule() {
            Rule::field_list => {
                FieldValue::UnparsedObject{field_list: field_value_raw, opt_type: field_type}
            },
            Rule::boolean => {
                panic!("bool not implemented");
            },
            Rule::float => {
                FieldValue::Float(field_value_raw.as_str().trim_end_matches("f").parse::<f32>()?)
            },
            Rule::string => {
                FieldValue::String(field_value_raw.into_inner().next().unwrap().as_str().to_string())
            },
            Rule::int => {
                FieldValue::Int(field_value_raw.as_str().trim_end_matches("i").parse::<i32>()?)
            },
            _ => unreachable!()
        };
        // insert into map, if duplicate, return error
        let displaced_val = field_map.insert(field_name.as_str().to_string(), field_value);
        if displaced_val.is_some() {
            bail!("duplicated field in field list");
        }
    }

    Ok(field_map)
}