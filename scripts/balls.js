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

    world.gravity.y = 0; // top-down view

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
                        texture: 'assets/balls/inpaint.png'
                    }
                }
            }),
    ]);

    const friction = 0.3;
    const restitution = 0.05;
    const density = 3;

    // add bodies
    Composite.add(world, [
        Bodies.circle(338, 464, 45, {
            density: density,
            frictionAir: 0.0006,
            restitution: restitution,
            friction: friction,
            render: {
                sprite: {
                    texture: 'assets/balls/1.png'
                }
            }
        })
    ]);

    Composite.add(world, [
        Bodies.circle(53, 442, 45, {
            density: density,
            frictionAir: 0.0006,
            restitution: restitution,
            friction: friction,
            render: {
                sprite: {
                    texture: 'assets/balls/2.png'
                }
            }
        })
    ]);


    Composite.add(world, [
        Bodies.circle(133, 480, 45, {
            isStatic: true,
            render: {
                sprite: {
                    texture: 'assets/balls/3.png'
                }
            }
        })
    ]);


    Composite.add(world, [
        Bodies.circle(174, 78, 45, {
            density: density,
            frictionAir: 0.0006,
            restitution: restitution,
            friction: friction,
            render: {
                sprite: {
                    texture: 'assets/balls/4.png'
                }
            }
        })
    ]);


    Composite.add(world, [
        Bodies.circle(285, 158, 45, {
            density: density,
            frictionAir: 0.0006,
            restitution: restitution,
            friction: friction,
            render: {
                sprite: {
                    texture: 'assets/balls/5.png'
                }
            }
        })
    ]);


    Composite.add(world, [
        Bodies.circle(203, 367, 45, {
            density: density,
            frictionAir: 0.0006,
            restitution: restitution,
            friction: friction,
            render: {
                sprite: {
                    texture: 'assets/balls/6.png'
                }
            }
        })
    ]);


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
