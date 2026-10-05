# rsf_waypoint_manager

Waypoint navigation, editing, visualization, and recording nodes for the RSF robot.

The navigation launch starts `waypoint_core`, which accepts
`/waypoint_navigator/start` and loads the configured waypoint YAML file.

## Section settings

A waypoint may carry `speed_limit` (m/s, published on `/speed_limit`) and `parameters`
(`node: {name: value}`). They apply while heading to that waypoint and the previous values
are restored once it is reached.

```yaml
- x: 155.19
  y: 16.33
  yaw: -2.436
  speed_limit: 0.3
  parameters:
    /local_costmap/local_costmap:
      footprint_padding: 0.0
    /collision_monitor:
      scan.enabled: false
```
