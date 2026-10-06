# rsf_waypoint_manager

Waypoint navigation, editing, visualization, and recording nodes for the RSF robot.

The navigation launch starts `waypoint_core`, which accepts
`/waypoint_navigator/start` and loads the configured waypoint YAML file.

## Profiles

A waypoint may carry `profile: <name>`. Profiles are defined in the file given by the
`profiles_file` parameter as `profiles: {name: {speed_limit: <m/s>, parameters: {node: {name: value}}}}`.
The named profile is applied while heading to that waypoint and stays active until a waypoint
names another one; `default` restores the original values, as does any stop of the navigation.
The active name is published on `~/profile` and the speed limit on `/speed_limit`.
In RViz, right-click a waypoint marker and pick a name under `Profile` (`Inherit` removes the
key); `~/save_waypoint` writes it to the file.

```yaml
- x: 155.19
  y: 16.33
  yaw: -2.436
  profile: narrow_slow
```
