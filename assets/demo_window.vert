#version 450

layout(push_constant) uniform Push {
    vec2 resolution;
} pc;

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 v_color;

void main() {
    // `position` is in window pixels with y pointing down, which is the same
    // direction as Vulkan's NDC y axis, so it maps across directly. Do NOT apply
    // the OpenGL-style y negation here: in Vulkan that flips the whole image
    // upside down (floor at the top, ceiling at the bottom).
    vec2 ndc = position / pc.resolution * 2.0 - 1.0;
    gl_Position = vec4(ndc.x, ndc.y, 0.0, 1.0);
    v_color = color;
}
