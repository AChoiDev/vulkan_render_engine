#version 460

layout(location = 1) out vec2 oTexCoord;
layout(location = 2) out vec2 oQuadSize;

struct Quad {
    vec4 base_pos_size;
    vec4 start_end_coords;
};

layout(std140, set = 0, binding = 5) readonly buffer QuadBuffer {
    Quad obj[];
} gi_buffer;

// layout( push_constant ) uniform UIQuadJob {
//     uint quad_start_index;
//     uint quad_count;
// } job;

void main() {
    // Setup vertex position offset
    int v_index_offset = gl_VertexIndex % 6;
    vec2 size_multiplier = vec2(0.0, 0.0);
    // TODO: make this a lookup table
    if (v_index_offset < 3) {
        // triangle 1 case
        if (v_index_offset == 0) {
            size_multiplier = vec2(1.0, 1.0);
        } else if (v_index_offset == 1) {
            size_multiplier = vec2(0.0, 1.0);
        } else {
            size_multiplier = vec2(0.0, 0.0);
        }
    } else {
        // triangle 2 case
        if (v_index_offset == 3) {
            size_multiplier = vec2(1.0, 1.0);
        } else if (v_index_offset == 4) {
            size_multiplier = vec2(0.0, 0.0);
        } else {
            size_multiplier = vec2(1.0, 0.0);
        }
    }

    // Setup vertex position
    Quad gi = gi_buffer.obj[gl_VertexIndex / 6];
    vec2 quad_base_pos = gi.base_pos_size.xy;
    vec2 quad_size = gi.base_pos_size.zw;
    vec2 pos = quad_base_pos + size_multiplier * quad_size;
    gl_Position = vec4(pos, 0.0, 1.0);

    oTexCoord = mix(gi.start_end_coords.xy, gi.start_end_coords.zw, size_multiplier);
    oQuadSize = quad_size;
    // Setup texture coordinates
    // let atlas_u_width = ((quad.width / GLYPH_WIDTH_UI_SPACE) * FONT_TEXEL_SIZE as f32) / ATLAS_PIXEL_WIDTH as f32;
    // const QUAD_WIDTH_TO_U_WIDTH: f32 =  (FONT_TEXEL_SIZE as f32) / (GLYPH_WIDTH_UI_SPACE as f32 * ATLAS_PIXEL_WIDTH as f32);
    // vec2 quad_base_tex_coords = gi.start_end_coords.xy;
    // oTexCoord = gi_start_end_coords.xy + size_multiplier * vec2(quad_width_to_u_width, quad_width_to_u_width) * quad_size;
}