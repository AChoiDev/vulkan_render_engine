#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform MyUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} mubo;

layout(set = 0, binding = 1) uniform sampler texSampler;

struct MySSBObject {
    mat4 model;
    uvec4 data_one;
};

struct TextureGroupDesc {
    uint base_color;
    uint normal;
    uint emissive;
    uint metallic;

    uint roughness;
    uint occlusion;
    uint padding_a;
    uint padding_b;
};

layout(std140, set = 0, binding = 2) readonly buffer MySSBuffer {
    MySSBObject obj[];
} mssb;

// TODO: nonuniform qualifier for variable sized array
layout(set = 0, binding = 3) uniform texture2D texArray[1024];

layout(std140, set = 0, binding = 4) readonly buffer MyTexGroupBuffer {
    TextureGroupDesc obj[];
} tex_group_desc_buffer;

// Vertex Input Attributes
layout(location = 0) in vec3 oColor;
layout(location = 1) in vec2 oTexCoord;
layout(location = 2) in vec3 interp_vertex_normal;
layout(location = 3) flat in uint i_gl_InstanceID;
layout(location = 4) in vec3 interp_vertex_tangent;
layout(location = 5) in smooth mat3 fragTBN_view_space;

// Output
layout(location = 0) out vec4 finalColor;


float extract_single_channel_component(uint tex_index_offset) {
    uint index_mask = 1073741823;

    uint tex_index = tex_index_offset & index_mask;
    vec4 texel = texture(sampler2D(texArray[tex_index], texSampler), oTexCoord);

    uint tex_offset = tex_index_offset >> 30;
    vec4 selection = vec4(equal(uvec4(tex_offset), uvec4(0, 1, 2, 3)));
    return dot(texel, selection);
}

vec3 visual_integer(uint value) {
    return (sin((vec3(value) + vec3(20.0, 12.0, 16.0)) * vec3(100.0)) * 0.5) + vec3(0.5);
}

void main() {

    uint texture_group_index = mssb.obj[i_gl_InstanceID].data_one.x;
    TextureGroupDesc tex_group = tex_group_desc_buffer.obj[texture_group_index];

    // setup base color, convert to linear rgb space
    vec3 vertex_normal = normalize(interp_vertex_normal);
    // vec3 vertex_tangent = normalize(interp_vertex_tangent);
    vec3 tex_base_color = pow(vec3(texture(sampler2D(texArray[tex_group.base_color], texSampler), oTexCoord)), vec3(2.2));

    finalColor = vec4(oTexCoord, 0.0, 1.0);
    finalColor = vec4(0.6, 0.3, 0.2, 1.0);
    finalColor = vec4(tex_base_color, 1.0);
    finalColor = vec4((vertex_normal * 0.5) + vec3(0.5, 0.5, 0.5), 1.0);
    finalColor = vec4(tex_base_color, 1.0);
    /*
    // setup texture normal
    uint normal_tex_index = tex_group.normal;
    vec2 unsigned_tex_normal_xy = vec2(texture(sampler2D(texArray[normal_tex_index], texSampler), oTexCoord));
    vec2 tex_normal_xy = (unsigned_tex_normal_xy - vec2(0.5)) * 2.0;
    float z_normal = sqrt(1.0 - dot(tex_normal_xy, tex_normal_xy));
    vec3 tex_normal = normalize(vec3(tex_normal_xy, z_normal));
    // tex_normal = vec3(0.0, 0.0, 1.0);
    vec3 normal_view_space = fragTBN_view_space * tex_normal;


    // setup emissive color
    uint emissive_tex_index = tex_group.emissive;
    vec3 tex_emissive = vec3(texture(sampler2D(texArray[emissive_tex_index], texSampler), oTexCoord));

    // setup single channels
    float tex_roughness = extract_single_channel_component(tex_group.roughness);
    float tex_metallic = extract_single_channel_component(tex_group.metallic);
    float tex_occlusion = extract_single_channel_component(tex_group.occlusion);

    // tex_occlusion = extract_single_channel_component(tex_group.roughness);


    
    // if (otbo == 3) {
    //     thingy = vec4(0.0, 1.0, 1.0, 0.0);
    // }

    // float tex_occulusion = dot(rgba_tex_occulusion, vec4(0.0, 0.0, 0.0, 1.0));

    // vec3 normal = vec3(tex_normal, 0.0);

    // finalColor = vec4(oColor, 1.0);
    // finalColor = vec4(oTexCoord, 0.0, 1.0);

    // vec3 base_color = vec3(texture(texSampler, oTexCoord) * vec4(oColor, 1.0));

    // vec3 lightDir = normalize(vec3(1.2, 0.7, 1.0));
    vec3 light_direction_world_space = normalize(vec3(2.0, 20.0, 7.0));
    vec3 light_direction_view_space = vec3(mubo.view * vec4(light_direction_world_space, 0.0));

    float dot_light = clamp(dot(normal_view_space, light_direction_view_space), 0.0, 1.0) * 0.8 + 0.2;
    // vec3 normal = oNormal;
    // float lambertian = max(dot(normal, lightDir), 0.0);
    // float intensity = lambertian * 0.8 + 0.2;


    // vec3 base_color = vec3(texture(sampler2D(texArray[texture_index], texSampler), oTexCoord));

    // finalColor = vec4(tex_color.xyz, 1.0);
    // finalColor = vec4(oTexCoord.xy, 0.0, 1.0);
    // finalColor = vec4((oNormal * 0.5) + vec3(0.5), 1.0);
    // finalColor = vec4(intensity * base_color, 1.0);

    vec3 vertex_normal = normalize(interp_vertex_normal);
    vec3 vertex_tangent = normalize(interp_vertex_tangent);

    uint visual_idx = (i_gl_InstanceID) % 4;
    vec3 visual_normal = (normal_view_space + 1.0) * 0.5;
    vec3 visual_tangent = (vertex_tangent + 1.0) * 0.5;

    if (visual_idx == 0) {
        finalColor = vec4(visual_normal, 1.0);
    } else if (visual_idx == 1) {
        finalColor = vec4(tex_base_color, 1.0);

    } else if (visual_idx == 2) {
        // finalColor = vec4(tex_base_color, 1.0);
        float occ = tex_occlusion * 0.5 + 0.4;
        finalColor = vec4(occ * visual_normal, 1.0);
    } else {
        finalColor = vec4(vec3(tex_roughness) * 0.5 + 0.4, 1.0);
    }

    finalColor = vec4(vec3(0.5), 1.0);
    finalColor = vec4(visual_integer(i_gl_InstanceID), 1.0);
    finalColor = vec4(tex_normal, 1.0);
    finalColor = vec4(visual_tangent, 1.0);
    finalColor = vec4(vec3(visual_tangent.y), 1.0);
    finalColor = vec4(visual_normal, 1.0);
    finalColor = vec4(vec3(dot_light), 1.0);
    finalColor = vec4(dot_light * tex_base_color, 1.0);
    finalColor = vec4((tex_normal + 1.0) * 0.5, 1.0);
    finalColor = vec4(tex_base_color, 1.0);
    finalColor = vec4(pow(dot_light * tex_base_color, vec3(1.0 / 2.2)), 1.0);
    // finalColor = vec4(vec3(1.0), 1.0);
    // finalColor = vec4(fragTBN())
    // i_gl_InstanceID
    */
}
