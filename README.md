## SUMMARY

We are going to implement a hydraulic erosion model for procedural terrain generation to produce more natural looking landscapes in real time.

## BACKGROUND
Procedurally generated heightmaps have been used to quickly create large height maps for video game terrain for decades. However, purely procedural techniques ultimately lack the natural qualities present in real landscapes.  A common method for increasing the realism of terrain is to model hydraulic erosion; the simulation of the flow of water removing and transporting surface material. However, modelling erosion accurately requires performing calculations on every cell of the heightmap being eroded for many iterations. This presents a significant bottleneck for CPU programs and a significant opportunity for parallelization with GPU’s.

## CHALLENGE

### The Problem
Hydraulic erosion is a process that has dependencies on : 
- Real-time computation
- Variable height
- Layers in the map that could erode
- Water erosion coefficients
Constructing this and rendering it will introduce potential constraints on parallelism that will need to be worked around.  

### Computation Time
The goal of this project is to develop an algorithm that can produce substantial hydraulic erosion on a procedurally generated map quickly enough for it to be included in the runtime of a video game. We are targeting real-time execution, meaning the algorithm can take at most a few milliseconds to render a given area and provide a performance of about 20 FPS.

## RESOURCES
The goal of the project is to provide a new, useful algorithm for video games, and therefore we are targeting consumer-level - CUDA capable hardware. We will use a 3rd party platform to render the output of our algorithm into a terrain mesh. 

## GOALS AND DELIVERABLES

### Goals
- Render a 1024x1024 heightmap in no more than 5ms per cycle
- Capable of operating on continuously scrolling terrain instead of a fixed heightmap
- Multiple terrain materials with different erosion mechanics
- Persistent rivers/bodies of water

### Stretch Goals
- Non-hydraulic erosion methods
- Voxel-based erosion instead of heightmap erosion

### Deliverables
- Working CUDA code and loading-unloading program
- Documentation in the form of a Report.

## PLATFORM CHOICE
We plan to make this project usable for video games and so will be using some ubiquitous NVIDIA hardware that is available to us. The specifics are:
- GeForce® GTX 1080 Ti 
- GeForce® GTX 1050 Ti 

## POSTER SESSION
Poster demonstrating the algorithm implemented and a live demo on one of our laptops showing the algorithm in process and iteration times. 

## SCHEDULE
| Description   | Date        |
|---------------|:-----------:|
| Determine algorithms for base heightmap generation, height map rendering environment. | Nov 05, 2018 |
| Implement basic height map generation algorithm. Design algorithm for hydraulic erosion. | Nov 12, 2018 |
| Begin implementation of an algorithm for hydraulic erosion. | Nov 19, 2018 |
| Finish implementation of the algorithm. | Nov 26, 2018 |
| Debug and optimize. Consider adding new features. | Dec 2, 2018 |
| Debug and optimize. Consider creating an interactive demonstration. | Dec 9, 2018 |
| Write the final report and prepare for the presentation. | Dec 16, 2018 |






