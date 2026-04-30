import pymunk
import math

class Ladder:
    def __init__(self, space, x, y, width, height):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = (x, y)
        self.shape = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.sensor = True  
        self.shape.filter = pymunk.ShapeFilter(categories=0b1000) 
        space.add(self.body, self.shape)
        self.rect = (x - width/2, y - height/2, x + width/2, y + height/2)

class Environment:
    def __init__(self):
        self.CAT_WALL  = 0b0001
        self.CAT_AGENT = 0b0010
        self.CAT_BOX   = 0b0100
        self.CAT_BALL  = 0b10000 
        self.ALL_MASKS = 0xFFFFFFFF 

        self.space = pymunk.Space()
        self.space.gravity = (0, -9.8)
        self.space.damping = 0.99 

        self.static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.space.add(self.static_body)

        WALL_R = 2.0 
        walls = [
            pymunk.Segment(self.static_body, (-12, -WALL_R), (12, -WALL_R), WALL_R),   
            pymunk.Segment(self.static_body, (-10-WALL_R, 0), (-10-WALL_R, 20), WALL_R), 
            pymunk.Segment(self.static_body, (10+WALL_R, 0), (10+WALL_R, 20), WALL_R),   
            pymunk.Segment(self.static_body, (-12, 15+WALL_R), (12, 15+WALL_R), WALL_R), 
        ]
        for wall in walls:
            wall.friction = 0.8
            wall.elasticity = 0.3
            wall.filter = pymunk.ShapeFilter(categories=self.CAT_WALL)
            self.space.add(wall)

        self.platforms = []
        def create_platform(x, y, w, h):
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (x, y)
            shape = pymunk.Poly.create_box(body, (w, h))
            shape.friction = 0.9
            shape.filter = pymunk.ShapeFilter(categories=self.CAT_WALL)
            self.space.add(body, shape)
            self.platforms.append((body, w, h))

        create_platform(3.0, 4.0, 2.0, 0.5)
        create_platform(5.5, 5.5, 2.0, 0.5)
        create_platform(8.0, 7.0, 2.0, 0.5)

        

        self.updraft_zone = (0.5, 2.5)  
        self.updraft_force = 22.0       

        self.water_level = 0.5
        self.in_water = False

        self.AGENT_MASS = 1.0
        self.AGENT_WIDTH = 0.8
        self.AGENT_HEIGHT_STAND = 1.6
        self.AGENT_HEIGHT_CROUCH = 0.8
        
        self.facing = 1 

        agent_moment = pymunk.moment_for_box(self.AGENT_MASS, (self.AGENT_WIDTH, self.AGENT_HEIGHT_STAND))
        self.agent_body = pymunk.Body(self.AGENT_MASS, agent_moment)
        self.agent_body.position = (-2.0, 1.5)

        self.agent_shape = pymunk.Poly.create_box(self.agent_body, (self.AGENT_WIDTH, self.AGENT_HEIGHT_STAND))
        self.agent_shape.friction = 1.0
        self.agent_shape.elasticity = 0.0
        self.agent_shape.filter = pymunk.ShapeFilter(categories=self.CAT_AGENT)
        self.space.add(self.agent_body, self.agent_shape)

        self.is_crouching = False
        self.is_sprinting = False
        self.is_climbing = False
        self.is_sleeping = False
        self.is_sitting = False # --- NEW SITTING STATE ---
        self.current_ladder = None
        self.is_rolling = False
        self.roll_timer = 0.0
        self.push_pull_object = None
        self.push_joint = None
        
        self.carried_body = None
        self.carried_shape = None
        self.carried_cat = 0
        self.grab_joint = None
        self.rotation_joint = None
        self.upright_torque_strength = 500.0

        self.BOX_SIZE = (0.6, 0.6)
        box_moment = pymunk.moment_for_box(2.0, self.BOX_SIZE)
        self.box_body = pymunk.Body(2.0, box_moment)
        self.box_body.position = (2.0, 0.8)
        self.box_shape = pymunk.Poly.create_box(self.box_body, self.BOX_SIZE)
        self.box_shape.friction = 0.7
        self.box_shape.elasticity = 0.2
        self.box_shape.filter = pymunk.ShapeFilter(categories=self.CAT_BOX)
        self.space.add(self.box_body, self.box_shape)

        self.BALL_RADIUS = 0.3
        ball_mass = 0.5
        ball_moment = pymunk.moment_for_circle(ball_mass, 0, self.BALL_RADIUS)
        self.ball_body = pymunk.Body(ball_mass, ball_moment)
        self.ball_body.position = (0.0, 1.0)
        self.ball_shape = pymunk.Circle(self.ball_body, self.BALL_RADIUS)
        self.ball_shape.friction = 0.6
        self.ball_shape.elasticity = 0.8
        self.ball_shape.filter = pymunk.ShapeFilter(categories=self.CAT_BALL) 
        self.space.add(self.ball_body, self.ball_shape)

        self.ladders = [Ladder(self.space, -5, 2, 1.0, 4.0)]
        self.bench_pos = (7, 1)

    # ==========================================
    # SENSORS & STATE
    # ==========================================
    def is_grounded(self):
        if self.is_climbing or self.is_sitting: return False
        angle = self.agent_body.angle
        px, py = self.agent_body.position
        half_h = self.agent_height / 2
        for offset in (-0.3, 0.0, 0.3):
            wx = offset * math.cos(angle)
            wy = offset * math.sin(angle)
            hits = self.space.segment_query(
                (px + wx, py + wy - half_h + 0.05), 
                (px + wx, py + wy - half_h - 0.3), 
                0.05, pymunk.ShapeFilter(mask=self.CAT_WALL)
            )
            for hit in hits:
                return True
        return False

    def can_stand_up(self):
        if not self.is_crouching: return True
        px, py = self.agent_body.position
        hits = self.space.segment_query(
            (px, py + (self.AGENT_HEIGHT_CROUCH / 2)), 
            (px, py + (self.AGENT_HEIGHT_STAND / 2) + 0.2), 
            0.05, pymunk.ShapeFilter(mask=self.CAT_WALL)
        )
        for hit in hits:
            return False
        return True

    def near_ladder(self):
        for ladder in self.ladders:
            x, y = self.agent_body.position
            if (ladder.rect[0] <= x <= ladder.rect[2] and ladder.rect[1] <= y <= ladder.rect[3]):
                return ladder
        return None

    @property
    def agent_height(self):
        return self.AGENT_HEIGHT_CROUCH if self.is_crouching else self.AGENT_HEIGHT_STAND

    # ==========================================
    # ACTIONS
    # ==========================================
    def toggle_sleep(self):
        if not self.is_grounded() or self.is_climbing or self.is_rolling: return False
        self.is_sleeping = not self.is_sleeping
        if self.is_sleeping:
            if not self.is_crouching: self.crouch()
            self.agent_body.velocity = (0, 0)
            self.is_sitting = False
        else:
            self.stand()
        return True

    def sit(self):
        # Toggle sit off if already sitting
        if self.is_sitting:
            self.is_sitting = False
            return True
            
        dist = math.dist((self.agent_body.position.x, self.agent_body.position.y), self.bench_pos)
        if dist < 1.5 and not self.is_sleeping:
            self.is_sitting = True
            self.is_crouching = False
            self.is_sleeping = False
            self.agent_body.velocity = (0, 0)
            self.agent_body.angular_velocity = 0
            # Snap exactly to bench seat coordinates
            self.agent_body.position = (self.bench_pos[0], self.bench_pos[1] + 0.55)
            # Face right to match the bench perspective
            self.facing = 1
            return True
        return False

    def walk_right(self):
        if self.is_sleeping: return
        self.is_sitting = False # Walking breaks sit state
        self.facing = 1 
        if self.is_climbing: return 
        
        vx, vy = self.agent_body.velocity
        if vx < (7.0 if self.is_sprinting else 5.0):
            force = 120.0 if self.is_sprinting else 80.0
            if not self.is_grounded(): force *= 0.3
            elif self.is_crouching: force *= 0.5
            self.agent_body.apply_force_at_world_point((force, 0), self.agent_body.position)

    def walk_left(self):
        if self.is_sleeping: return
        self.is_sitting = False # Walking breaks sit state
        self.facing = -1 
        if self.is_climbing: return 
        
        vx, vy = self.agent_body.velocity
        if vx > -(7.0 if self.is_sprinting else 5.0):
            force = -120.0 if self.is_sprinting else -80.0
            if not self.is_grounded(): force *= 0.3
            elif self.is_crouching: force *= 0.5
            self.agent_body.apply_force_at_world_point((force, 0), self.agent_body.position)

    def jump(self):
        if self.is_sleeping: return False
        self.is_sitting = False # Jumping breaks sit state
        if self.is_climbing or self.is_crouching: return False
        if self.is_grounded() and self.agent_body.velocity.y <= 0.5:
            self.agent_body.velocity = (self.agent_body.velocity.x, 0)
            self.agent_body.apply_impulse_at_world_point((0, 10.0), self.agent_body.position)
            return True
        return False

    def crouch(self):
        if self.is_sleeping: return False
        self.is_sitting = False
        if self.is_climbing: return False
        if not self.is_crouching and self.is_grounded():
            self.is_crouching = True
            self.space.remove(self.agent_shape)
            self.agent_shape = pymunk.Poly.create_box(self.agent_body, (self.AGENT_WIDTH, self.agent_height))
            self.agent_shape.friction = 1.0
            self.agent_shape.elasticity = 0.0
            self.agent_shape.filter = pymunk.ShapeFilter(categories=self.CAT_AGENT)
            self.space.add(self.agent_shape)
            self.agent_body.moment = pymunk.moment_for_box(self.AGENT_MASS, (self.AGENT_WIDTH, self.agent_height))
            return True
        return False

    def stand(self):
        if self.is_sleeping: return False
        if self.is_crouching and self.can_stand_up():
            self.is_crouching = False
            self.space.remove(self.agent_shape)
            self.agent_shape = pymunk.Poly.create_box(self.agent_body, (self.AGENT_WIDTH, self.agent_height))
            self.agent_shape.friction = 1.0
            self.agent_shape.elasticity = 0.0
            self.agent_shape.filter = pymunk.ShapeFilter(categories=self.CAT_AGENT)
            self.space.add(self.agent_shape)
            self.agent_body.moment = pymunk.moment_for_box(self.AGENT_MASS, (self.AGENT_WIDTH, self.agent_height))
            return True
        return False

    def start_sprint(self):
        if not self.is_crouching and self.is_grounded(): self.is_sprinting = True
    def stop_sprint(self): self.is_sprinting = False

    def throw_or_kick(self):
        if self.is_sleeping or self.is_sitting: return False
        if self.carried_body is not None:
            body_to_throw = self.carried_body
            self.release_object()
            body_to_throw.velocity = (self.facing * 6.0, 2.0)
            return True
            
        dist_box = math.dist((self.agent_body.position.x, self.agent_body.position.y), (self.box_body.position.x, self.box_body.position.y))
        if dist_box < 2.0:
            self.box_body.velocity = (self.facing * 10.0, 1.0)
            return True

        dist_ball = math.dist((self.agent_body.position.x, self.agent_body.position.y), (self.ball_body.position.x, self.ball_body.position.y))
        if dist_ball < 1.5:
            self.ball_body.velocity = (self.facing * 8.0, 2.0)
            return True
        return False

    def grab_object(self):
        if self.is_sleeping or self.is_sitting: return False
        if self.carried_body is not None or self.push_pull_object is not None: return False
        
        dist_box = math.dist((self.agent_body.position.x, self.agent_body.position.y), (self.box_body.position.x, self.box_body.position.y))
        dist_ball = math.dist((self.agent_body.position.x, self.agent_body.position.y), (self.ball_body.position.x, self.ball_body.position.y))

        target_body, target_shape, cat = None, None, 0

        if dist_ball < 2.0 and dist_ball <= dist_box:
            target_body, target_shape, cat = self.ball_body, self.ball_shape, self.CAT_BALL
        elif dist_box < 2.0:
            target_body, target_shape, cat = self.box_body, self.box_shape, self.CAT_BOX

        if target_body is not None:
            target_shape.filter = pymunk.ShapeFilter(categories=cat, mask=self.ALL_MASKS ^ self.CAT_AGENT)
            self.grab_joint = pymunk.PivotJoint(self.agent_body, target_body, (self.facing * 0.8, 0.2), (0, 0))
            self.rotation_joint = pymunk.GearJoint(self.agent_body, target_body, 0.0, 1.0)
            self.space.add(self.grab_joint, self.rotation_joint)
            
            self.carried_body = target_body
            self.carried_shape = target_shape
            self.carried_cat = cat
            return True
        return False

    def release_object(self):
        if self.carried_body is not None:
            self.space.remove(self.grab_joint, self.rotation_joint)
            self.grab_joint = self.rotation_joint = None
            self.carried_shape.filter = pymunk.ShapeFilter(categories=self.carried_cat)
            self.carried_body = None
            self.carried_shape = None

    def start_push_pull(self):
        if self.is_sleeping or self.is_sitting: return False
        if self.carried_body is not None or self.push_pull_object is not None: return False
        dist = math.dist((self.agent_body.position.x, self.agent_body.position.y), (self.box_body.position.x, self.box_body.position.y))
        if dist < 2.0:
            self.push_pull_object = self.box_body
            self.push_joint = pymunk.SlideJoint(self.agent_body, self.box_body, (0,0), (0,0), 0.5, 1.5)
            self.space.add(self.push_joint)
            return True
        return False

    def stop_push_pull(self):
        if self.push_pull_object is not None:
            self.space.remove(self.push_joint)
            self.push_joint = self.push_pull_object = None

    def start_climb(self, ladder):
        if self.is_sleeping or self.is_sitting: return False
        if self.is_climbing: return False
        self.is_climbing = True
        self.current_ladder = ladder
        self.agent_body.velocity = (0, 0)
        self.space.gravity = (0, 0) 
        return True

    def stop_climb(self):
        if self.is_climbing:
            self.is_climbing = False
            self.current_ladder = None
            self.space.gravity = (0, -9.8)

    def move_on_ladder(self, vx, vy):
        if not self.is_climbing or not self.current_ladder: return
        self.agent_body.velocity = (vx, vy)
        l, b, r, t = self.current_ladder.rect
        px, py = self.agent_body.position
        clamped_x = max(l, min(px, r))
        clamped_y = max(b, min(py, t)) 
        if px != clamped_x or py != clamped_y:
            self.agent_body.position = (clamped_x, clamped_y)

    def roll(self):
        if self.is_sleeping or self.is_sitting: return False
        if self.is_rolling or self.is_climbing or not self.is_grounded(): return False
        self.agent_body.apply_impulse_at_world_point((self.facing * 600, 200), self.agent_body.position)
        self.agent_body.angular_velocity = 10.0
        self.is_rolling = True
        self.roll_timer = 0.5 
        self.agent_shape.filter = pymunk.ShapeFilter(categories=self.CAT_AGENT, mask=self.ALL_MASKS ^ self.CAT_BOX ^ self.CAT_BALL)
        return True

    def update_physics(self, dt):
        
        for body in [self.agent_body, self.box_body, self.ball_body]:
            px, py = body.position
            
            if py < self.water_level and 2.0 <= px <= 9.0:
                body.velocity = (body.velocity.x * 0.95, body.velocity.y * 0.95)
                body.angular_velocity *= 0.95
                
            if self.updraft_zone[0] < px < self.updraft_zone[1] and py < 6.0:
                lift_force = 40.0 * body.mass 
                body.apply_force_at_world_point((0, lift_force), body.position)
                if body.velocity.y > 5.0:
                    body.velocity = (body.velocity.x, 5.0)

        # --- NEW: LOCK AGENT IN PLACE WHILE SITTING ---
        if self.is_sitting:
            self.agent_body.position = (self.bench_pos[0], self.bench_pos[1] + 0.55)
            self.agent_body.velocity = (0, 0)
            self.agent_body.angular_velocity = 0
            self.agent_body.angle = 0
        
        self.in_water = (self.agent_body.position.y < self.water_level) and (2.0 <= self.agent_body.position.x <= 9.0)

        if not self.is_climbing and not self.is_rolling and not self.is_sitting:
            angle = self.agent_body.angle
            while angle > math.pi: angle -= 2 * math.pi
            while angle < -math.pi: angle += 2 * math.pi
            self.agent_body.torque = -self.upright_torque_strength * angle - 20.0 * self.agent_body.angular_velocity

        if self.is_rolling:
            self.roll_timer -= dt
            if self.roll_timer <= 0.0:
                self.is_rolling = False
                self.agent_shape.filter = pymunk.ShapeFilter(categories=self.CAT_AGENT)

        if self.carried_body is not None and self.grab_joint is not None:
            self.grab_joint.anchor_a = (self.facing * 0.8, 0.2)

        if self.carried_body is None and self.push_pull_object is None:
            self.box_body.angular_velocity *= 0.97
            self.ball_body.angular_velocity *= 0.99

        vx, vy = self.agent_body.velocity
        max_speed = 10.0 if self.is_sprinting else 8.0
        speed = math.sqrt(vx*vx + vy*vy)
        if speed > max_speed:
            self.agent_body.velocity = (vx * max_speed / speed, vy * max_speed / speed)

        for _ in range(3): self.space.step(dt / 3)

    def get_state(self):
        ax, ay = self.agent_body.position
        return f"Agent({ax:.1f}, {ay:.1f})"