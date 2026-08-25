# InstinctLab-specific Instructions

## Common Practice

- Following the common practice in the pinned IsaacLab.
- For class registry, if using `str` instead of `type` as the key, using `{DIR}` is encourged.

## Sensor and Multi-Backend support

Since currently IsaacLab supports both Newton and PhysX backends, some scene entity should support different backend implementation. Typical scene entity implemented in this repository shall have base interface and backend implementation.

- Each backend-releated scene entity should have a base interface class, which is backend-agnostic, named by the entity name, e.g. `VolumePointsBase`, `MotionReferenceManagerBase`.

- In the scene entity folder, the backend implementation should be placed in their dedicated files, e.g. `volume_points/newton.py`, `volume_points/physx.py`. The backend implementation class should be named by the entity name with the backend name as suffix, e.g. `VolumePointsNewton`, `VolumePointsPhysX`.

- Using dynamic backend dispatch logic to select which backend implementation to use, e.g. `VolumePoints` class will dispatch to `VolumePointsNewton` or `PhysXVolumePoints` based on the current backend.

### Grouped Ray-Caster and Noisy Camera

- They are the extension for multiple IsaacLab's builtin sensors. So, use Mixin implementation is encouraged instead of following the base interface and backend implementation pattern.

## Monitors

### Sensor-typed Monitors

- Please also implement the `Base` class and the auto backend-dispatch class in @instinctlab/monitors/monitors.py and implement those backend-dependent implementation in @instinctlab/monitors/newton.py and @instinctlab/monitors/physx.py respectively.
