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

    world.gravity.y = 1;

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
    Runner.run(runner, engine);

    world.bodies = [];

    // these static walls will not be rendered in this sprites example, see options
    
    Composite.add(world, [
        Bodies.rectangle(256, 256, 512, 512,
            {
                isStatic: true,
                isSensor: true,
                render: {
                    sprite: {
                        texture: 'assets/domino/inpaint.png'
                    }
                }
            }),
    ]);

    const friction = 1.0;
    const restitution = 0.01;
    const density = 0.01;

    Composite.add(world, [
        Bodies.rectangle(37, 394, 22, 123, {
                density: density,
                frictionAir: 0.0006,
                restitution: restitution,
                friction: friction,
                render: {
                    sprite: {
                        texture: 'assets/domino/1.png'
                    }
                }
            })
    ]);


    Composite.add(world, [
        Bodies.rectangle(121, 394, 22, 123, {
                density: density,
                frictionAir: 0.0006,
                restitution: restitution,
                friction: friction,
                render: {
                    sprite: {
                        texture: 'assets/domino/2.png'
                    }
                }
            })
    ]);

    Composite.add(world, [
        Bodies.rectangle(205, 394, 22, 123, {
                density: density,
                frictionAir: 0.0006,
                restitution: restitution,
                friction: friction,
                render: {
                    sprite: {
                        texture: 'assets/domino/3.png'
                    }
                }
            })
    ]);

    Composite.add(world, [
        Bodies.rectangle(289, 394, 22, 123, {
                density: density,
                frictionAir: 0.0006,
                restitution: restitution,
                friction: friction,
                render: {
                    sprite: {
                        texture: 'assets/domino/4.png'
                    }
                }
            })
    ]);

    Composite.add(world, [
        Bodies.rectangle(373, 394, 22, 123, {
                density: density,
                frictionAir: 0.0006,
                restitution: 0.01,
                friction: friction,
                render: {
                    sprite: {
                        texture: 'assets/domino/5.png'
                    }
                }
            })
    ]);

    Composite.add(world, [
        Bodies.rectangle(457, 394, 22, 123, {
                density: density,
                frictionAir: 0.0006,
                restitution: 0.01,
                friction: friction,
                render: {
                    sprite: {
                        texture: 'assets/domino/6.png'
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

    var offset = 60;
    createInvisibleEdge(-offset, 455.5, 512+offset, 455.5);
    
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
