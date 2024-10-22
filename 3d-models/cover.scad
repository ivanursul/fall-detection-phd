// Original Box Parameters
length = 76;            // Outer length of the box
width = 37;             // Outer width of the box
height = 25;            // Height of the box
wall_thickness = 2;     // Thickness of the walls
corner_radius = 5;      // Radius of the outer corners

// Derived Box Parameters
inner_length = length - 2 * wall_thickness; // 70 mm
inner_width = width - 2 * wall_thickness;   // 32 mm
inner_corner_radius = corner_radius - wall_thickness; // 3 mm

// Tolerance for proper fitting
fit_tolerance = 0.2;    // Clearance for the lip to fit into the box

// Cover Parameters
cover_thickness = 2;    // Thickness of the cover's top
lip_height = 2;         // Height of the lip that goes into the box
lip_length = 72 - 2 * fit_tolerance;  // Adjusted for fit tolerance
lip_width = 33 - 2 * fit_tolerance;   // Adjusted for fit tolerance
lip_corner_radius = inner_corner_radius - fit_tolerance; // Adjusted corner radius

// Generate the Cover with Lip
module cover_with_lip() {
    union() {
        // Top plate of the cover
        translate([0, 0, height])
            linear_extrude(cover_thickness)
                rounded_rectangle(length, width, corner_radius);
        
        // Lip that fits inside the box
        translate([0, 0, height - lip_height])
            linear_extrude(lip_height)
                rounded_rectangle(
                    lip_length,
                    lip_width,
                    lip_corner_radius >= 0 ? lip_corner_radius : 0
                );
    }
}

// Render the Cover with Lip
cover_with_lip();

// Module to create a rounded rectangle
module rounded_rectangle(len, wid, rad) {
    rad = min(rad, len / 2, wid / 2);
    offset(r = rad)
        square([len - 2 * rad, wid - 2 * rad], center = true);
}
