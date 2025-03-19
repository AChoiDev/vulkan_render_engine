use std::collections::HashMap;

use winit::event::ElementState;

use crate::{id_display, ui_tree, RuntimeNode, RuntimeNodeID, TextTypeInput};


pub fn handle_console(input_events: Vec<winit::event::KeyEvent>, 
    console_open: &mut bool, 
    console_input: &mut String, 
    console_back_lines: &mut Vec<String>, 
    runtime_scene: &generational_arena::Arena<RuntimeNode>, 
    time_of_time_start: &mut std::time::Instant, 
    ui_nodes: &mut Vec<ui_tree::UiNode>) {
    // process inputs for console
    for input_event in input_events {
        // TODO: use result types for console commands

        use winit::keyboard::*;
        let is_toggle_console_input = 
            input_event.state == ElementState::Pressed
            && input_event.repeat == false
            && input_event.physical_key == PhysicalKey::Code(KeyCode::Backquote);

        if is_toggle_console_input
        {
            // toggle the console visibility
            // must early terminate to not pass toggle input to console text input
            *console_open = !*console_open;
            continue;
        }

        // early terminate if input is not a text input
        let type_input = match TextTypeInput::new(&input_event) {
            Some(val) => val,
            None => continue,
        };

        match type_input {
            TextTypeInput::Graphic(ch) => {
                console_input.push(ch);
            }
            TextTypeInput::Backspace => {
                // remove the rightmost character if present
                if console_input.len() > 0 {
                    *console_input =
                        console_input[0..console_input.len() - 1].to_string();
                }
            }
            TextTypeInput::Enter => {
                // print console input to back lines
                console_back_lines.push(format!("] {}", console_input.clone()));

                // get copy of console input and clear input out
                let input_line = console_input.clone();
                let input_parts: Vec<_> = input_line.split_whitespace().collect();
                *console_input = "".to_string();

                let &cmd_name = match input_parts.first() {
                    Some(val) => val,
                    None => continue,
                };
                
                let cmd_params = &input_parts[1..];

                if cmd_name == "echo" {
                    // echo to console
                    console_back_lines.push(cmd_params.join(" "));
                } 
                else if cmd_name == "node" {
                    // echo out scene node
                    use regex::Regex;
                    let re = Regex::new(r"^i(\d+)g(\d+)$").unwrap();
                    if cmd_params.len() == 1 {
                        if re.is_match(cmd_params[0]) {
                            let node_id_str = cmd_params[0];
                            let g_char_idx = node_id_str.find("g").unwrap();
                            let node_index = node_id_str[1..g_char_idx].parse::<usize>().unwrap();
                            let gen_index = node_id_str[g_char_idx + 1..].parse::<u64>().unwrap();
                            let node_id = generational_arena::Index::from_raw_parts(node_index, gen_index);
                            if let Some(node) = runtime_scene.get(node_id) {
                                console_back_lines.push(format!("{:?}", node));
                            } else {
                                console_back_lines.push("Node not found".to_string());
                            }
                        } else {
                            console_back_lines.push("Invalid node id format".to_string());
                        }
                    } else {
                        console_back_lines.push("Invalid number of parameters for node command".to_string());
                    }
                    // re.is_match_at()
                    // use std::string::re
                    // if command_params
                }
                else if cmd_name == "scene" {
                    // echo out scene tree
                    console_back_lines.push("Scene Tree:".to_string());

                    // obtain children map and root nodes
                    let mut children_map = HashMap::<RuntimeNodeID, Vec<RuntimeNodeID>>::new();
                    let mut root_nodes = Vec::<RuntimeNodeID>::new();
                    runtime_scene.iter().for_each(|(id, node)| {
                        if let Some(parent_id) = node.opt_parent_node_id {
                            children_map
                                .entry(parent_id)
                                .or_insert(Vec::new())
                                .push(id);
                        } else {
                            // root node
                            root_nodes.push(id);
                        }
                    });

                    // push root nodes onto stack
                    let mut stack = Vec::<(RuntimeNodeID, usize)>::new();
                    for root_node_id in root_nodes.iter().rev() {
                        stack.push((*root_node_id, 0));
                    }

                    // traverse nodes
                    while let Some((node_id, depth)) = stack.pop() {
                        let mut str = "  ".to_string();
                        for _ in 0..depth {
                            str.push_str("|-- ");
                        }
                        let id_string = id_display(node_id);
                        if let Some(name) = &runtime_scene.get(node_id).unwrap().opt_name {
                            // str.push_str(id_display(node_id).as_str());
                            str.push_str(format!("{} ({})", name, id_string).as_str());
                        } else {
                            str.push_str(id_string.as_str());
                        }
                        console_back_lines.push(str);

                        if let Some(children) = children_map.get(&node_id) {
                            for child_id in children.iter().rev() {
                                stack.push((*child_id, depth + 1));
                            }
                        }
                    }
                } 
                else if cmd_name == "set_time" {
                    let &time_str = match cmd_params.first() {
                        Some(val) => val,
                        None => {
                            console_back_lines.push("Invalid number of parameters for command".to_string());
                            continue
                        },
                    };

                    match time_str.parse::<f32>() {
                        Ok(time) => {
                            if time >= 0.0 {
                                *time_of_time_start = std::time::Instant::now() - std::time::Duration::from_secs_f32(time);
                            } else {
                                console_back_lines.push("Negative time not allowed.".to_string());
                            }
                        },
                        Err(err) => {
                            console_back_lines.push("Could not parse float".to_string());
                            console_back_lines.push(err.to_string())
                        },
                    }
                }
                else {
                    let str =
                        format!("Unknown Command \"{}\"", cmd_name);
                    console_back_lines.push(str);
                }
            }
            // TextTypeInput::ArrowDown => todo!(),
            // TextTypeInput::ArrowLeft => todo!(),
            // TextTypeInput::ArrowRight => todo!(),
            // TextTypeInput::ArrowUp => todo!(),
            _ => {}
        }

    }

    if *console_open == false
    {
        // early terminate to prevent drawing the console UI
        return;
    }

    // setup console ui
    use ui_tree::UiContent;
    use ui_tree::UiNode;

    // background rectangle
    ui_nodes.push(UiNode {
        content: UiContent::FilledColor {
            color: [0.05, 0.05, 0.12, 0.8],
        },
        spacing: ui_tree::Positioning {
            x: -1.0,
            y: -1.0,
            width: 2.0,
            height: 1.05,
        },
    });


    // create console string output text
    let mut console_str = "".to_string();
    for line in console_back_lines.iter() {
        console_str.push_str(line);
        console_str.push('\n');
    }
    console_str.push_str(format!("] {}", console_input).as_str());
    ui_nodes.push(UiNode {
        content: UiContent::Text {
            color: [1.0, 1.0, 1.0, 1.0],
            font_size: 0.1,
            text_str: console_str.clone(),
        },
        spacing: ui_tree::Positioning {
            x: -1.0,
            y: -1.0,
            width: 2.0,
            height: 1.0,
        },
    });

}
