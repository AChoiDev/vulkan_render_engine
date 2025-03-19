use std::collections::HashMap;

struct NormalizedGlyphMetrics {
    // in [0, 1] range
    atlas_uv: [f32 ; 2],

    // offset from glyph origin to top-left corner of quad in units of font size
    offset_x: f32,
    offset_y: f32,
    advance_x: f32,
    
    // size of quad in units of font size
    size_x: f32,
    size_y: f32,
}



impl NormalizedGlyphMetrics {
    fn new(metrics: fontsdf::Metrics, atlas_uv: [f32 ; 2], font_texel_size: f32) -> Self {
        let size_y = metrics.height as f32 / font_texel_size;
        // TODO: find an easier way to explain this equation
        let offset_y = -(size_y + (metrics.bounds.ymin * 2.0 - metrics.ymin as f32) as f32 / font_texel_size);
        Self {
            atlas_uv,
            offset_x: (metrics.xmin as f32) / font_texel_size,
            offset_y,
            advance_x: metrics.advance_width as f32 / font_texel_size,
            size_x: (metrics.width as f32) / font_texel_size,
            size_y,
        }
    }
}

pub struct QuadFont {
    glyphs: HashMap<char, NormalizedGlyphMetrics>,
    normalized_space_advance: f32,
    pub atlas: Vec<u8>, // RGBA square image
    atlas_texel_width: u32,
    font_texel_size: f32,
}

// todo: clean this code
impl QuadFont {
    pub fn new(font: fontsdf::Font, font_texel_size: u32, atlas_texel_width: u32) -> Self {

        let needed_graphics_chars_set: Vec<char>= ('!'..='~').into_iter().collect();

        // determine the locations to write the glyphs on the atlas
        let mut img_y_curr = 0u32; // the top pixel of the current glyph row being written to
        let mut img_y_max = 0u32; // the most deepest pixel written to along the y axis
        let mut img_x_curr = 0u32; // the next pixel to write to along the x axis
        let mut n_glyphs = HashMap::new();
        let mut out_glyphs: Vec<(Vec<u8>, fontsdf::Metrics, (u32, u32))> = Vec::new();
        for c in needed_graphics_chars_set.iter() {
            let (metrics, sdf) = font.rasterize_sdf(*c, font_texel_size as f32);

            if metrics.width as u32 > atlas_texel_width {
                panic!("Character too wide for image");
            }
            if metrics.height as u32 > atlas_texel_width {
                panic!("Character too tall for image");
            }

            // check if character can fit in current glyph row
            let x_leftover = atlas_texel_width as i32 - img_x_curr as i32;
            if metrics.width as i32 > x_leftover {
                // character is too wide for current row
                // move to next glyph row
                img_y_curr = img_y_max + 1;
                img_x_curr = 0;
            }

            // character now has enough space along the x-axis

            // check if there's enough vertical space in row
            let y_leftover = atlas_texel_width as i32 - img_y_curr as i32;
            if metrics.height as i32 > y_leftover {
                // character is too tall for current row
                // no more space in image, throw error
                panic!("Character too tall for image");
            }

            // character now has enough space along the y-axis
            // write along row
            if *c == ',' || *c == '.' {
                println!("glyph: {:?}", metrics);
            }
            
            let atlas_uv = [img_x_curr as f32 / atlas_texel_width as f32, img_y_curr as f32 / atlas_texel_width as f32];
            let odd_metric = NormalizedGlyphMetrics::new(metrics, atlas_uv, font_texel_size as f32);
            n_glyphs.insert(*c, odd_metric);
            out_glyphs.push((sdf, metrics, (img_x_curr as u32, img_y_curr as u32)));
            img_x_curr += metrics.width as u32;
            img_y_max = (img_y_max as i32).max(img_y_curr as i32 + (metrics.height as i32 - 1)) as u32;
        }

        let mut atlas: Vec<u8> = Vec::new();
        for _ in 0..(atlas_texel_width * atlas_texel_width * 4) {
            atlas.push(0);
        }

        // write glyph bitmaps to atlas
        for (glyph_bitmap, metrics, table_loc) in out_glyphs.iter() {
            for (g_p_idx, glyph_pixel) in glyph_bitmap.iter().enumerate() {
                let g_p_x = g_p_idx as i32 % metrics.width as i32;
                let g_p_y = g_p_idx as i32 / metrics.width as i32;

                let table_pixel_x = table_loc.0 as i32 + g_p_x;
                let table_pixel_y = table_loc.1 as i32 + g_p_y;

                let table_pixel_idx = (table_pixel_y * (atlas_texel_width as i32) + table_pixel_x) * 4;
                atlas[table_pixel_idx as usize] = *glyph_pixel;
            }
        }
        let normalized_space_advance = font.metrics(' ', font_texel_size as f32, true).advance_width / (font_texel_size as f32);

        Self {
            glyphs: n_glyphs,
            normalized_space_advance,
            atlas,
            atlas_texel_width: atlas_texel_width as u32,
            font_texel_size: font_texel_size as f32,
        }

    }
}

// A quad in UI space representing a glyph instance
#[derive(Debug)]
pub struct GlyphQuad {
    // in UI space units
    pub x: f32,
    pub y: f32,
    pub height: f32,
    pub width: f32,

    // in atlas uv units
    pub atlas_u: f32,
    pub atlas_v: f32,
}

pub enum TextVerticalAlignment {
    Top, Center, Bottom
}

pub struct PixelSnapping {
    pub pixel_width: u32,
    pub pixel_height: u32,
}

// todo: make this code more clean
// glyph density: the number of glyphs per unit of ui space
pub fn generate_quads(font: &QuadFont, input_text: String, text_box_origin: [f32 ; 2], 
    text_box_size: [f32 ; 2], glyph_width_in_ui_space: f32, glyph_height_in_ui_space: f32,
    text_vertical_alignment: TextVerticalAlignment, opt_snapping: Option<PixelSnapping>)
-> Vec<GlyphQuad> {
    let mut out_quads = Vec::new();

    // Split text into chunks
    struct GlyphChunk {
        start_char_idx: usize,
        char_count: usize,
        advance_width: f32,
    }
    enum TextChunk {
        Glyphs(GlyphChunk),
        Space,
        Tab,
        Newline,
    }
    let mut text_chunks = Vec::new();
    let mut opt_curr_word: Option<GlyphChunk> = None;
    let mut previous_char_was_mid_split = false;
    for (i, c) in input_text.chars().enumerate() {
        
        match c {
            // whitespace case
            ' ' | '\t' | '\n' | '\r' => {
                if let Some(word) = opt_curr_word.take() {
                    text_chunks.push(TextChunk::Glyphs(word));
                }

                // handle specific white case types
                match c {
                    ' ' => text_chunks.push(TextChunk::Space),
                    '\t' => text_chunks.push(TextChunk::Tab),
                    '\n' => text_chunks.push(TextChunk::Newline),
                    '\r' => text_chunks.push(TextChunk::Newline),
                    _ => panic!("Unexpected character"),
                }
            },
            // handle graphical glyph case
            _ => {
                let metric = font.glyphs.get(&c).unwrap();
                if previous_char_was_mid_split && c != '-' {
                    if let Some(word) = opt_curr_word.take() {
                        text_chunks.push(TextChunk::Glyphs(word));
                    }
                }
                let curr_word = opt_curr_word.get_or_insert(GlyphChunk {
                    start_char_idx: i,
                    char_count: 0,
                    advance_width: 0.0,
                });
                curr_word.advance_width += metric.advance_x * glyph_width_in_ui_space;
                curr_word.char_count += 1;
            },
        }

        if c == '-' {
            previous_char_was_mid_split = true;
        } else {
            previous_char_was_mid_split = false;
        }
    }
    if let Some(word) = opt_curr_word.take() {
        text_chunks.push(TextChunk::Glyphs(word));
    }

    // word wrap the glyph chunks for rendering
    let mut glyph_chunks_to_render = Vec::new();
    let line_count;
    let line_height_in_ui_space: f32 = glyph_height_in_ui_space * 1.4f32;
    {
        let char_space_size_in_ui_space = font.normalized_space_advance * glyph_width_in_ui_space;
        let mut x_curr = text_box_origin[0];
        let max_x = x_curr + text_box_size[0];
        let mut y_line_idx = 0;
        // let mut y_curr = text_box_origin[1] + line_height_in_ui_space;
        for text_chunk in text_chunks {
            match text_chunk {
                TextChunk::Glyphs(word) => {
                    let advance_width = word.advance_width;
                    let leftover = max_x - x_curr;
                    if advance_width > leftover {
                        x_curr = text_box_origin[0];
                        y_line_idx += 1;
                    }
                    glyph_chunks_to_render.push((word, (x_curr, y_line_idx)));
                    x_curr += advance_width;
                },
                TextChunk::Space => {
                    x_curr += char_space_size_in_ui_space;
                },
                TextChunk::Tab => {
                    // x_curr += font_tab_width;
                    x_curr += char_space_size_in_ui_space * 4.0;
                },
                TextChunk::Newline => {
                    x_curr = text_box_origin[0];
                    y_line_idx += 1;
                },
            }
        }

        line_count = y_line_idx + 1;
    }

    let y_start =
        match text_vertical_alignment {
            TextVerticalAlignment::Top => {
                text_box_origin[1]
            },
            TextVerticalAlignment::Center => {
                text_box_origin[1] + (text_box_size[1] - (line_height_in_ui_space * line_count as f32)) / 2.0
            },
            TextVerticalAlignment::Bottom => {
                text_box_origin[1] + text_box_size[1] - (line_height_in_ui_space * line_count as f32)
            },
        };
    
    
    // produce glyph quads to render
    for (glyph_chunk, (chunk_x_pos, chunk_y_line_idx)) in glyph_chunks_to_render.iter() {
        let mut x_from = *chunk_x_pos;
        let chunk_y_pos = (chunk_y_line_idx + 1) as f32 * line_height_in_ui_space + y_start;
        for i in glyph_chunk.start_char_idx..(glyph_chunk.start_char_idx + glyph_chunk.char_count) {
            // let metric = odd_metrics.iter().find(|m| m.character == text[i]).unwrap();
            let c = input_text.chars().nth(i).unwrap();
            let metric = font.glyphs.get(&c).unwrap();
            let ui_width = glyph_width_in_ui_space as f32;
            let ui_height = glyph_height_in_ui_space as f32;

            let base_x = x_from + metric.offset_x * ui_width;
            let base_y = chunk_y_pos + metric.offset_y * ui_height;

            let (x, y) =
                if let Some(snapping) = &opt_snapping {
                    // map (-1, 1) to (0, width)
                    let pix_width = snapping.pixel_width as f32;
                    let res_x = (base_x + 1.0) * 0.5f32 * (snapping.pixel_width as f32);
                    let rounded_x = res_x.round();
                    // map (0, width) to (-1, 1)
                    let snapped_x = (rounded_x / pix_width) * 2.0 - 1.0;

                    let pix_height = snapping.pixel_height as f32;
                    let res_y = (base_y + 1.0) * 0.5f32 * (snapping.pixel_height as f32);
                    let rounded_y = res_y.round();
                    let snapped_y = (rounded_y / pix_height) * 2.0 - 1.0;

                    (snapped_x, snapped_y)
                } else {
                    (base_x, base_y)
                };

            // let atlas_uv = metric.opt_atlas_uv.unwrap();
            let quad = GlyphQuad {
                x,
                y,
                width: metric.size_x * ui_width,
                height: metric.size_y * ui_height,
                atlas_u: metric.atlas_uv[0],
                atlas_v: metric.atlas_uv[1],
            };
            out_quads.push(quad);
            x_from += metric.advance_x * ui_width;
        }
    }

    out_quads
}