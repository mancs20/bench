include "globals.mzn";

var bool: minimization;
int: num_images;
int: universe;
int: max_cloud_area;

set of int: IMAGES = 1..num_images; %IMAGES
set of int: UNIVERSE = 1..universe;

array[IMAGES] of set of int: images;
array[IMAGES] of set of int: clouds;
array[IMAGES] of int: costs;
% array[IMAGES] of int: resolution;
array[IMAGES] of int: incidence_angle;
array[UNIVERSE] of int: areas;

array[IMAGES] of var bool: decision_vars; % take the image i or not

% Set covering constraint.
constraint forall(u in UNIVERSE)(
  exists(i in IMAGES)(decision_vars[i] /\ u in images[i]));

% weighed set covering constraint
var int: total_cost = sum(i in IMAGES)(costs[i] * decision_vars[i]);

% var int: max_resolution = sum(u in UNIVERSE)(min(i in IMAGES where u in images[i] /\ decision_vars[i])(resolution[i]));

var int: max_incidence = max(i in IMAGES)(decision_vars[i] * incidence_angle[i]);

% CLOUDS CONSTRAINTS
% Which images have a universe `u` without cloud?
% That is, uclear[u] = {i1, i2, ..} means that the images numbered i1, i2, ... contains `u` without clouds.
array[UNIVERSE] of set of int: uclear = [{ i | i in IMAGES where not (u in clouds[i]) /\ u in images[i] } | u in UNIVERSE];
% cloudy[u] is true iff no image containing a version of `u` without clouds is taken.
array[UNIVERSE] of var bool: cloudy;
array[UNIVERSE] of var int: num_clear_images;
constraint forall(u in UNIVERSE)(
  num_clear_images[u] = sum(i in uclear[u])(decision_vars[i])
);
constraint forall(u in UNIVERSE)(cloudy[u] = (num_clear_images[u] == 0));

var int: cloudy_area = sum(u in UNIVERSE)(cloudy[u] * areas[u]);



array[1..3] of var int: objs;
constraint objs[1] = total_cost;
constraint objs[2] = cloudy_area;
% constraint objs[3] = max_resolution;
constraint objs[3] = max_incidence;

% array[1..4] of var bool: minimize_objs;
% array[1..4] of var int: ref_point;

% constraint ref_point[1] = sum(i in IMAGES)(costs[i]) + 1;
% constraint ref_point[2] = sum(u in UNIVERSE)(areas[u]) + 1;
% constraint ref_point[3] = sum(u in UNIVERSE)(max(i in IMAGES where u in images[i])(resolution[i])) + 1;
% constraint ref_point[4] = 900;

% constraint minimize_objs[1] = true;
% constraint minimize_objs[2] = true;
% constraint minimize_objs[3] = true;
% constraint minimize_objs[4] = true;

solve satisfy;
