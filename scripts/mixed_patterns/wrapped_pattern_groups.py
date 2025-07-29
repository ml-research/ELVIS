# Created by MacBook Pro at 21.07.25


from scripts.proximity import util_red_triangle, util_fixed_props, util_big_small, util_grid_objs, util_weird_circle
from scripts.similarity import util_fixed_number, util_pacman, util_palette
from scripts.closure import util_feature_circle, util_feature_square, util_feature_triangle, util_pos_triangle, util_pos_circle, util_pos_square
from scripts.symmetry import util_solar_system, util_symmetry_cir
from scripts.continuity import util_a_splines, util_one_split_n, util_two_splines, util_u_splines, util_x_feature_splines


# Proximity wrappers
def wrap_non_overlap_red_triangle(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_red_triangle.non_overlap_red_triangle(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin)


def wrap_non_overlap_fixed_props(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_fixed_props.non_overlap_fixed_props(fixed_props, is_positive, obj_quantities, pin)


def wrap_overlap_big_small(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_big_small.overlap_big_small(fixed_props, is_positive, cluster_num, obj_quantities, pin)


def wrap_non_overlap_grid(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_grid_objs.non_overlap_grid(fixed_props, is_positive, cluster_num, obj_quantities, pin)


def wrap_overlap_circle_features(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_weird_circle.overlap_circle_features(fixed_props, is_positive, cluster_num, obj_quantities, 1, pin)


# Similarity wrappers
def wrap_non_overlap_fixed_number(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_fixed_number.non_overlap_fixed_number(fixed_props, is_positive, cluster_num, obj_quantities)


def wrap_non_overlap_pacman(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_pacman.non_overlap_pacman(fixed_props, is_positive, cluster_num, obj_quantities)


def wrap_non_overlap_palette(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_palette.non_overlap_palette(fixed_props, is_positive, cluster_num, obj_quantities)


# Closure wrappers
def wrap_non_overlap_feature_circle(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_feature_circle.non_overlap_feature_circle(fixed_props, is_positive, cluster_num, pin)


def wrap_non_overlap_feature_square(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_feature_square.non_overlap_feature_square(fixed_props, is_positive, cluster_num, pin)


def wrap_non_overlap_feature_triangle(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_feature_triangle.non_overlap_feature_triangle(fixed_props, is_positive, cluster_num, pin)


def wrap_non_overlap_big_triangle(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_pos_triangle.separate_big_triangle(fixed_props, is_positive, cluster_num, obj_quantities, pin)


def wrap_non_overlap_big_circle(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_pos_circle.non_overlap_big_circle(fixed_props, is_positive, cluster_num, obj_quantities, pin)


def wrap_non_overlap_big_square(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_pos_square.separate_big_square(fixed_props, is_positive, cluster_num, obj_quantities, pin)


# Symmetry wrappers
def wrap_non_overlap_solar_sys(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_solar_system.non_overlap_soloar_sys(fixed_props, is_positive, cluster_num)


def wrap_feature_symmetry_circle(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_symmetry_cir.feature_symmetry_circle(fixed_props, is_positive, cluster_num)


# Continuity wrappers
def wrap_non_overlap_a_splines(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_a_splines.non_overlap_a_splines(fixed_props, is_positive, cluster_num, obj_quantities, pin)


def wrap_non_overlap_one_split_n(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_one_split_n.non_overlap_one_split_n(fixed_props, is_positive, cluster_num, obj_quantities, pin)


def wrap_non_overlap_two_splines(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_two_splines.non_overlap_two_splines(fixed_props, is_positive, cluster_num, obj_quantities, pin)


def wrap_non_overlap_u_splines(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_u_splines.non_overlap_u_splines(fixed_props, is_positive, cluster_num, obj_quantities, pin)


def wrap_feature_continuity_x_splines(fixed_props, is_positive, cluster_num, obj_quantities, qualifiers, pin):
    return util_x_feature_splines.feature_continuity_x_splines(fixed_props, is_positive, cluster_num, obj_quantities, pin)
