pub struct Positioning {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}


// TODO: make this a macro
pub enum UiContent {
    Text { 
        text_str: String, color: [f32 ; 4], font_size: f32, 
    },
    FilledColor { 
        color: [f32 ; 4], 
    },
}

pub struct UiNode {
    pub content: UiContent,
    pub spacing: Positioning,
}

// impl UiNode {
//     pub fn spacing(&self) -> &Positioning {
//         match self {
//             UiNode::UiGroup { spacing, .. } => spacing,
//             UiNode::UiText { spacing, .. } => spacing,
//             UiNode::UiFilledColor { spacing, .. } => spacing,
//         }
//     }

//     pub fn children(&self) -> &Vec<UiNode> {
//         match self {
//             UiNode::UiGroup { children, .. } => children,
//             UiNode::UiText { children, .. } => children,
//             UiNode::UiFilledColor { children, .. } => children,
//         }
//     }
// }
