var Example = Example || {};


Example.physgen = function () {
    var Engine = Matter.Engine,
        Render = Matter.Render,
        Runner = Matter.Runner,
        Composites = Matter.Composites,
        Common = Matter.Common,
        MouseConstraint = Matter.MouseConstraint,
        Mouse = Matter.Mouse,
        Vertices = Matter.Vertices,
        Composite = Matter.Composite,
        Bodies = Matter.Bodies;

    Body = Matter.Body;

    // create engine
    var engine = Engine.create(),
        world = engine.world;

    world.gravity.y = 9.8;

    // create renderer

    var render = Render.create({
        element: document.body,
        engine: engine,
        options: {
            width: 512,
            height: 512,
            showAngleIndicator: false,
            wireframes: false
        }
    });

    Render.run(render);

    // create runner
    var runner = Runner.create();
    //Runner.run(runner, engine);

    // Set up the fixed time step (e.g., 1000 / 120 = 120 FPS)
    // Increase the number of position and velocity iterations
    engine.positionIterations = 10;  // Default is 6
    engine.velocityIterations = 10;  // Default is 4

    var fixedDelta = 1000 / 120;
    // Custom runner to manually control the time step
    (function run() {
        window.requestAnimationFrame(run);
        Runner.tick(runner, engine, fixedDelta); // This will update the engine with a fixed time step
        Render.world(render); // Render the scene
    })();

    world.bodies = [];

    Composite.add(world, [
        Bodies.rectangle(256, 256, 512, 512,
            {
                isStatic: true,
                isSensor: true,
                render: {
                    sprite: {
                        texture: 'assets/pig_ball/inpaint.png'
                    }
                }
            }),
    ]);

    Composite.add(world, [
        Bodies.circle(256, 126, 61, {
            density: 0.1,
            frictionAir: 0.0006,
            restitution: 0.65,
            friction: 0.7,
            render: {
                sprite: {
                    texture: 'assets/pig_ball/1.png'
                }
            }
        })
    ]);

    Composite.add(world, [
        Bodies.rectangle(445, 165, 130, 34, {
                isStatic: true,
                render: {
                    sprite: {
                        texture: 'assets/pig_ball/2.png'
                    }
                }
            })
    ]);


    const points2 = [[471, 100], [470, 101], [469, 101], [469, 102], [467, 104], [463, 104], [462, 105], [461, 105], [460, 104], [458, 104], [457, 103], [451, 103], [449, 105], [449, 110], [448, 111], [443, 111], [442, 110], [439, 110], [437, 112], [436, 112], [433, 115], [433, 117], [432, 118], [432, 120], [431, 121], [431, 138], [432, 139], [432, 140], [433, 141], [433, 143], [436, 146], [443, 146], [444, 145], [448, 145], [449, 144], [450, 144], [451, 145], [452, 145], [453, 146], [454, 146], [455, 147], [456, 147], [458, 149], [458, 150], [460, 152], [476, 152], [478, 150], [479, 150], [481, 152], [497, 152], [498, 151], [498, 145], [499, 144], [499, 140], [498, 139], [498, 137], [499, 136], [499, 135], [500, 134], [500, 132], [501, 131], [501, 121], [500, 120], [500, 118], [499, 117], [499, 116], [498, 115], [498, 114], [496, 112], [496, 111], [495, 110], [494, 110], [491, 107], [490, 107], [489, 106], [488, 106], [487, 105], [485, 105], [484, 104], [480, 104], [478, 102], [477, 102], [476, 101], [475, 101], [474, 100]];
    const vertices2 = points2.map(point => ({ x: point[0], y: point[1]}));
    Composite.add(world, [
        Bodies.fromVertices(467, 127, vertices2, {
            density: 1.5,
            frictionAir: 0.0006,
            restitution: 0.1,
            friction: 1.5,
            render: {
                sprite: {
                    texture: 'assets/pig_ball/3.png'
                }
            }
        })
    ]);

    function createInvisibleEdge(x1, y1, x2, y2, thickness=5, direction = "center") {
        // Calculate the center position of the edge
        let centerX = (x1 + x2) / 2;
        let centerY = (y1 + y2) / 2;
    
        // Calculate the length and angle of the edge
        const length = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
        const angle = Math.atan2(y2 - y1, x2 - x1);
    
        // Adjust the center position based on the direction and thickness
        switch (direction.toLowerCase()) {
            case "top":
                centerX += (thickness / 2) * Math.sin(angle);
                centerY -= (thickness / 2) * Math.cos(angle);
                break;
            case "bottom":
                centerX -= (thickness / 2) * Math.sin(angle);
                centerY += (thickness / 2) * Math.cos(angle);
                break;
            case "left":
                centerX -= (thickness / 2) * Math.cos(angle);
                centerY -= (thickness / 2) * Math.sin(angle);
                break;
            case "right":
                centerX += (thickness / 2) * Math.cos(angle);
                centerY += (thickness / 2) * Math.sin(angle);
                break;
            default:
                // Default is center, no change needed
                break;
        }
    
        // Create a rectangle for the edge with the specified thickness
        const edge = Bodies.rectangle(centerX, centerY, length, thickness, {
            isStatic: true, // Make it static so it doesn't move
            render: {
                visible: false // Make it invisible
            }
        });
    
        // Set the angle of the rectangle to match the endpoints
        Body.setAngle(edge, angle);
    
        // Add the edge to the world
        Composite.add(world, edge);
    
        return edge;
    }
    
    createInvisibleEdge(409, 152, 512, 152);
    createInvisibleEdge(182, 187, 512, 187);
    
    // var offset = 10;
    //createInvisibleEdge(-offset, -offset, 512+offset, -offset); // top
    //createInvisibleEdge(-offset, -offset, -offset, 512+offset); // left
    createInvisibleEdge(-20, 512-20, 512+20, 512-20); // bottom
    //createInvisibleEdge(512+offset, -offset, 512+offset, 512+offset); // right

    
    // add mouse control
    var mouse = Mouse.create(render.canvas),
        mouseConstraint = MouseConstraint.create(engine, {
            mouse: mouse,
            constraint: {
                stiffness: 0.2,
                render: {
                    visible: false
                }
            }
        });

    Composite.add(world, mouseConstraint);

    // keep the mouse in sync with rendering
    render.mouse = mouse;

    // fit the render viewport to the scene
    Render.lookAt(render, {
        min: { x: 0, y: 0 },
        max: { x: 512, y: 512 }
    });

    // context for MatterTools.Demo
    return {
        engine: engine,
        runner: runner,
        render: render,
        canvas: render.canvas,
        stop: function () {
            Matter.Render.stop(render);
            Matter.Runner.stop(runner);
        }
    };
};

Example.physgen.title = 'Physgen';
Example.physgen.for = '>=0.14.2';

if (typeof module !== 'undefined') {
    module.exports = Example.physgen;
}
