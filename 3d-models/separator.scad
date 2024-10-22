// Box parameters (should match your existing box parameters)
length = 76;             // Outer length of the box
width = 37;              // Outer width of the box
wall_thickness = 2;      // Thickness of the walls
corner_radius = 5;       // Radius of the outer corners

// Separator parameters
separator_thickness = 1;           // Thickness of the separator
separator_clearance = 0.2;         // Clearance for proper fit

// Pillar parameters
pillar_radius = 1.5;               // Radius of the pillars
pillar_height = 14;                // Height of the pillars (as per your request)
pillar_margin = 1;                 // Margin from the edges

// Ventilation parameters
vent_line_width = 1;             // Width of the ventilation lines (made thinner)
vent_line_spacing = 3;             // Spacing between lines (center to center)

// Inner dimensions of the box
inner_length = length - 2 * wall_thickness;
inner_width  = width - 2 * wall_thickness;

// Inner corner radius
inner_corner_radius = corner_radius - wall_thickness;

// Separator dimensions
separator_length = inner_length - 2 * separator_clearance;
separator_width  = inner_width - 2 * separator_clearance;
separator_radius = inner_corner_radius - separator_clearance;

// Create the separator with ventilation lines and pillars
module separator_with_ventilation_lines_and_pillars() {
    // Separator plate with ventilation lines
    difference() {
        // Separator plate
        translate([0, 0, 0])
            linear_extrude(separator_thickness)
                rounded_rectangle(separator_length, separator_width, separator_radius);

        // Ventilation lines
        vent_lines();
    }

    // Pillars
    // Positions for the pillars (stepped back by pillar_margin from the edges)
    positions = [
        [ separator_length / 2 - pillar_margin - pillar_radius,  separator_width / 2 - pillar_margin - pillar_radius],
        [-separator_length / 2 + pillar_margin + pillar_radius,  separator_width / 2 - pillar_margin - pillar_radius],
        [-separator_length / 2 + pillar_margin + pillar_radius, -separator_width / 2 + pillar_margin + pillar_radius],
        [ separator_length / 2 - pillar_margin - pillar_radius, -separator_width / 2 + pillar_margin + pillar_radius]
    ];

    // Create pillars at each corner, connected to the separator
    for (pos = positions) {
        translate([pos[0], pos[1], -pillar_height])
            pillar(pillar_radius, pillar_height + separator_thickness);
    }
}

// Module to create a rounded rectangle
module rounded_rectangle(len, wid, rad) {
    rad = min(rad, len / 2, wid / 2); // Ensure radius does not exceed half the length or width
    offset(r = rad)
        square([len - 2 * rad, wid - 2 * rad], center = true);
}

// Module to create a pillar (cylinder)
module pillar(radius, height) {
    cylinder(h = height, r = radius, center = false);
}

// Updated Module to create centered, thinner ventilation lines
module vent_lines() {
    // Define the area where lines can be placed, avoiding pillars and edges
    x_min = -separator_length / 2 + pillar_margin + pillar_radius + vent_line_width / 2;
    x_max = separator_length / 2 - pillar_margin - pillar_radius - vent_line_width / 2;
    y_min = -separator_width / 2 + pillar_margin + pillar_radius + vent_line_width / 2;
    y_max = separator_width / 2 - pillar_margin - pillar_radius - vent_line_width / 2;

    // Calculate the number of lines that fit within x_min and x_max
    available_width = x_max - x_min;
    num_lines = floor((available_width + vent_line_spacing) / vent_line_spacing);
    total_spacing = (num_lines - 1) * vent_line_spacing;
    offset = (available_width - total_spacing) / 2;

    // Create lines across the separator, centered
    for (i = [0 : num_lines - 1]) {
        x = x_min + offset + i * vent_line_spacing;
        translate([x - vent_line_width / 2, y_min, -1]) // Adjusted positioning
            cube([vent_line_width, y_max - y_min, separator_thickness + 2], center = false);
    }
}

// Render the separator with ventilation lines and pillars
separator_with_ventilation_lines_and_pillars();
