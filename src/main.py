"""Aurora Evolutions - an interactive 2D evolution sandbox.

The simulation renders a grid world populated by lifeforms with tiny genetic
matrices. Genes are mapped to behavioural traits including diet, environmental
preferences and reproduction behaviour. Players can paint different materials,
seed organisms and observe emergent ecosystems.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import pygame

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

CELL_SIZE = 18
GRID_WIDTH = 46
GRID_HEIGHT = 32
PANEL_HEIGHT = 180
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE + PANEL_HEIGHT
FPS = 60
SIMULATION_STEP_SECONDS = 0.35
MAX_LIFEFORM_COUNT = 1200

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


# ---------------------------------------------------------------------------
# Environment definitions
# ---------------------------------------------------------------------------


class Material(Enum):
    SOIL = auto()
    WATER = auto()
    ROCK = auto()
    SAND = auto()


@dataclass(frozen=True)
class MaterialProperties:
    color: Tuple[int, int, int]
    fertility: float
    light_multiplier: float
    heat_bias: float
    moisture: float


MATERIAL_PROPERTIES: Dict[Material, MaterialProperties] = {
    Material.SOIL: MaterialProperties(color=(92, 76, 55), fertility=1.0, light_multiplier=0.95, heat_bias=1.8, moisture=0.4),
    Material.WATER: MaterialProperties(color=(34, 90, 140), fertility=0.4, light_multiplier=0.75, heat_bias=-2.5, moisture=1.0),
    Material.ROCK: MaterialProperties(color=(120, 120, 126), fertility=0.15, light_multiplier=0.85, heat_bias=0.4, moisture=0.1),
    Material.SAND: MaterialProperties(color=(194, 172, 92), fertility=0.25, light_multiplier=1.05, heat_bias=3.0, moisture=0.2),
}


MATERIAL_ORDER = [Material.SOIL, Material.WATER, Material.ROCK, Material.SAND]


class DietType(Enum):
    PHOTOTROPH = auto()
    HERBIVORE = auto()
    CARNIVORE = auto()
    OMNIVORE = auto()


@dataclass
class Traits:
    diet: DietType
    preferred_temp: float
    temp_tolerance: float
    preferred_light: float
    light_tolerance: float
    preferred_material: Material
    metabolism: float
    reproduction_energy: float
    mutation_rate: float
    speed: int
    max_age: int
    diet_efficiency: float
    sociality: float


GeneMatrix = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


@dataclass
class Lifeform:
    identifier: int
    x: int
    y: int
    genes: GeneMatrix
    traits: Traits
    species_name: str
    color: Tuple[int, int, int]
    energy: float = 90.0
    age: int = 0
    alive: bool = True
    children: int = 0

    def position(self) -> Tuple[int, int]:
        return self.x, self.y


@dataclass
class SpeciesInfo:
    name: str
    genes: GeneMatrix
    traits: Traits
    color: Tuple[int, int, int]
    discovered_tick: int
    description: str
    population: int = 0
    historical_max: int = 0
    total_births: int = 0
    total_deaths: int = 0


# ---------------------------------------------------------------------------
# Genetic interpretation helpers
# ---------------------------------------------------------------------------


SPECIES_SYLLABLES = [
    "ae", "vi", "ra", "lo", "an", "ti", "mo", "sa", "zen", "qua", "ly", "sha", "er", "tu", "ka",
    "na", "re", "phi", "za", "dor", "wen", "ul", "zor", "el", "ia",
]


def normalise_matrix(matrix: Sequence[Sequence[float]]) -> GeneMatrix:
    rows: List[Tuple[float, float, float]] = []
    for row in matrix:
        rows.append((clamp(row[0], 0.0, 1.0), clamp(row[1], 0.0, 1.0), clamp(row[2], 0.0, 1.0)))
    return rows[0], rows[1], rows[2]


def genes_to_color(matrix: GeneMatrix) -> Tuple[int, int, int]:
    flattened = [int(value * 255) for row in matrix for value in row]
    r = (flattened[0] + flattened[3] + flattened[6]) // 3
    g = (flattened[1] + flattened[4] + flattened[7]) // 3
    b = (flattened[2] + flattened[5] + flattened[8]) // 3
    # soften brightness to keep colours visible on dark backgrounds
    r = clamp(r * 0.85 + 40, 30, 235)
    g = clamp(g * 0.85 + 40, 30, 235)
    b = clamp(b * 0.85 + 40, 30, 235)
    return int(r), int(g), int(b)


def matrix_signature(matrix: GeneMatrix) -> str:
    return "-".join(f"{value:.2f}" for row in matrix for value in row)


def matrix_to_species_name(matrix: GeneMatrix) -> str:
    seed_value = sum(int(value * 9973) for row in matrix for value in row)
    rng = random.Random(seed_value)
    syllables = [rng.choice(SPECIES_SYLLABLES) for _ in range(3)]
    base = "".join(syllables)
    return base.capitalize()


def interpret_genes(matrix: GeneMatrix) -> Traits:
    g = matrix
    diet_weights = [g[0][0], g[0][1], g[0][2]]
    highest = max(diet_weights)
    # determine if omnivorous by closeness of best two genes
    sorted_weights = sorted(diet_weights, reverse=True)
    if sorted_weights[0] - sorted_weights[1] < 0.12:
        diet = DietType.OMNIVORE
    else:
        diet = [DietType.PHOTOTROPH, DietType.HERBIVORE, DietType.CARNIVORE][diet_weights.index(highest)]

    preferred_temp = 6 + g[1][0] * 36
    temp_tolerance = 3 + g[2][2] * 7
    preferred_light = 0.25 + g[1][1] * 0.75
    light_tolerance = 0.1 + g[2][0] * 0.5
    material_index = int(g[1][2] * len(MATERIAL_ORDER)) % len(MATERIAL_ORDER)
    preferred_material = MATERIAL_ORDER[material_index]

    metabolism = 0.45 + g[2][1] * 0.6
    reproduction_energy = 80 + g[2][1] * 80
    mutation_rate = 0.015 + g[2][0] * 0.12
    speed = 1 + int(g[0][2] * 2.5)
    max_age = 160 + int(g[0][1] * 420)
    diet_efficiency = 0.55 + g[0][0] * 0.9
    sociality = g[1][0]

    return Traits(
        diet=diet,
        preferred_temp=preferred_temp,
        temp_tolerance=temp_tolerance,
        preferred_light=preferred_light,
        light_tolerance=light_tolerance,
        preferred_material=preferred_material,
        metabolism=metabolism,
        reproduction_energy=reproduction_energy,
        mutation_rate=mutation_rate,
        speed=speed,
        max_age=max_age,
        diet_efficiency=diet_efficiency,
        sociality=sociality,
    )


def traits_description(traits: Traits) -> str:
    diet_text = {
        DietType.PHOTOTROPH: "absorbs light",
        DietType.HERBIVORE: "grazes on vegetation",
        DietType.CARNIVORE: "hunts other species",
        DietType.OMNIVORE: "switches diet opportunistically",
    }[traits.diet]
    material = traits.preferred_material.name.title()
    return (
        f"Prefers {traits.preferred_temp:.0f}°C, light {traits.preferred_light:.2f}, "
        f"{diet_text} and favours {material.lower()} terrain"
    )


def generate_random_genes(target_diet: Optional[DietType] = None) -> GeneMatrix:
    matrix = [[random.random() for _ in range(3)] for _ in range(3)]
    if target_diet is not None:
        idx = [DietType.PHOTOTROPH, DietType.HERBIVORE, DietType.CARNIVORE].index(
            target_diet if target_diet != DietType.OMNIVORE else DietType.HERBIVORE
        )
        for i in range(3):
            matrix[0][i] *= 0.4
        matrix[0][idx] = clamp(random.uniform(0.6, 1.0), 0.0, 1.0)
        if target_diet == DietType.OMNIVORE:
            matrix[0][2] = clamp(random.uniform(0.4, 0.8), 0.0, 1.0)
    return normalise_matrix(matrix)


def mutate_genes(matrix: GeneMatrix, mutation_rate: float) -> GeneMatrix:
    mutated: List[List[float]] = [list(row) for row in matrix]
    for y in range(3):
        for x in range(3):
            base = mutated[y][x]
            delta = random.gauss(0, mutation_rate * 0.35)
            if random.random() < mutation_rate:
                delta += random.uniform(-0.2, 0.2)
            mutated[y][x] = clamp(base + delta, 0.0, 1.0)
    return normalise_matrix(mutated)


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------


class World:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.tiles: List[List[Material]] = [[Material.SOIL for _ in range(width)] for _ in range(height)]
        self.temperature: List[List[float]] = [[18.0 for _ in range(width)] for _ in range(height)]
        self.light: List[List[float]] = [[0.8 for _ in range(width)] for _ in range(height)]
        self.plants: List[List[float]] = [[random.random() * 0.5 for _ in range(width)] for _ in range(height)]
        self.tick: int = 0
        self.global_light: float = 0.8
        self.global_temp: float = 18.0
        self.day_phase: float = random.random() * math.tau

    def reset(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                self.tiles[y][x] = random.choice(MATERIAL_ORDER)
                self.temperature[y][x] = 18.0 + random.uniform(-2.0, 2.0)
                self.light[y][x] = 0.7 + random.uniform(-0.1, 0.1)
                self.plants[y][x] = random.random() * 0.4
        self.tick = 0
        self.day_phase = random.random() * math.tau

    def update_environment(self) -> None:
        self.tick += 1
        self.day_phase = (self.day_phase + 0.015) % math.tau
        sun_factor = max(0.2, math.sin(self.day_phase) * 0.8 + 0.2)
        self.global_light = clamp(sun_factor, 0.15, 1.0)
        self.global_temp = 12.0 + self.global_light * 16.0
        for y in range(self.height):
            for x in range(self.width):
                material = self.tiles[y][x]
                props = MATERIAL_PROPERTIES[material]
                target_temp = self.global_temp + props.heat_bias + (self.light[y][x] - 0.5) * 4
                self.temperature[y][x] += (target_temp - self.temperature[y][x]) * 0.08
                base_light = self.global_light * props.light_multiplier
                vegetation_block = self.plants[y][x] * 0.3
                self.light[y][x] = clamp(base_light - vegetation_block, 0.05, 1.1)
                fertility = props.fertility
                growth = fertility * self.light[y][x] * 0.05
                decay = (1 - fertility) * 0.02
                if material == Material.WATER:
                    growth *= 0.35
                    decay *= 1.4
                self.plants[y][x] = clamp(self.plants[y][x] + growth - decay, 0.0, 1.0)

    def set_material(self, x: int, y: int, material: Material) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.tiles[y][x] = material


class Simulation:
    def __init__(self) -> None:
        self.world = World(GRID_WIDTH, GRID_HEIGHT)
        self.lifeforms: List[Lifeform] = []
        self.species_registry: Dict[str, SpeciesInfo] = {}
        self.encyclopedia_scroll: int = 0
        self.selected_material: Material = Material.SOIL
        self.pending_births: List[Lifeform] = []
        self.time_accumulator: float = 0.0
        self.next_life_identifier: int = 1
        self.paused: bool = False

    # ------------------------------------------------------------------
    # Lifeform management helpers
    # ------------------------------------------------------------------

    def create_lifeform(self, x: int, y: int, genes: Optional[GeneMatrix] = None) -> Optional[Lifeform]:
        if not (0 <= x < self.world.width and 0 <= y < self.world.height):
            return None
        if len(self.lifeforms) + len(self.pending_births) >= MAX_LIFEFORM_COUNT:
            return None
        genes = genes or generate_random_genes()
        traits = interpret_genes(genes)
        species_name = matrix_to_species_name(genes)
        color = genes_to_color(genes)
        life = Lifeform(
            identifier=self.next_life_identifier,
            x=x,
            y=y,
            genes=genes,
            traits=traits,
            species_name=species_name,
            color=color,
            energy=random.uniform(70.0, 110.0),
        )
        self.next_life_identifier += 1
        self.register_species(life)
        self.lifeforms.append(life)
        return life

    def register_species(self, life: Lifeform) -> None:
        if life.species_name not in self.species_registry:
            description = traits_description(life.traits)
            self.species_registry[life.species_name] = SpeciesInfo(
                name=life.species_name,
                genes=life.genes,
                traits=life.traits,
                color=life.color,
                discovered_tick=self.world.tick,
                description=description,
            )
        species = self.species_registry[life.species_name]
        species.total_births += 1

    def record_death(self, life: Lifeform) -> None:
        if life.species_name in self.species_registry:
            self.species_registry[life.species_name].total_deaths += 1

    # ------------------------------------------------------------------
    # Simulation update logic
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        if self.paused:
            return
        self.time_accumulator += dt
        while self.time_accumulator >= SIMULATION_STEP_SECONDS:
            self.step()
            self.time_accumulator -= SIMULATION_STEP_SECONDS

    def step(self) -> None:
        self.world.update_environment()
        for birth in self.pending_births:
            self.lifeforms.append(birth)
        self.pending_births.clear()

        cell_map: Dict[Tuple[int, int], List[Lifeform]] = defaultdict(list)
        survivors: List[Lifeform] = []
        for life in self.lifeforms:
            if not life.alive:
                continue
            life.age += 1
            if life.age > life.traits.max_age:
                life.alive = False
                self.record_death(life)
                continue

            self.apply_environment_effects(life)
            if not life.alive:
                self.record_death(life)
                continue

            self.move_lifeform(life)
            if not life.alive:
                self.record_death(life)
                continue

            cell_map[life.position()].append(life)
            survivors.append(life)

        self.lifeforms = survivors
        self.handle_predation(cell_map)
        self.lifeforms = [life for life in self.lifeforms if life.alive]
        for life in self.lifeforms:
            self.try_reproduce(life)

        for birth in self.pending_births:
            self.register_species(birth)
        self.lifeforms.extend(self.pending_births)
        self.pending_births.clear()

        self.update_species_populations()

    def apply_environment_effects(self, life: Lifeform) -> None:
        x, y = life.position()
        temp = self.world.temperature[y][x]
        light_level = self.world.light[y][x]
        material = self.world.tiles[y][x]

        temp_diff = abs(temp - life.traits.preferred_temp)
        temp_penalty = max(0.0, temp_diff - life.traits.temp_tolerance) * 0.08
        light_diff = abs(light_level - life.traits.preferred_light)
        light_penalty = max(0.0, light_diff - life.traits.light_tolerance) * 1.2
        material_penalty = 0.35 if material != life.traits.preferred_material else -0.2

        life.energy -= (life.traits.metabolism + temp_penalty + light_penalty + material_penalty)
        life.energy = max(life.energy, 0.0)

        if life.traits.diet == DietType.PHOTOTROPH:
            gain = light_level * (2.2 + life.traits.diet_efficiency)
            life.energy += gain
        elif life.traits.diet == DietType.HERBIVORE:
            plants = self.world.plants[y][x]
            consume = min(plants, 0.22 * life.traits.diet_efficiency)
            self.world.plants[y][x] -= consume
            life.energy += consume * 130
        elif life.traits.diet == DietType.OMNIVORE:
            plants = self.world.plants[y][x]
            consume = min(plants, 0.12 * life.traits.diet_efficiency)
            self.world.plants[y][x] -= consume
            life.energy += consume * 80 + light_level * 1.2

        if life.energy <= 0:
            life.alive = False

    def evaluate_cell_suitability(self, life: Lifeform, x: int, y: int) -> float:
        temp = self.world.temperature[y][x]
        light_level = self.world.light[y][x]
        material = self.world.tiles[y][x]
        score = 0.0
        score -= abs(temp - life.traits.preferred_temp) * 0.08
        score -= abs(light_level - life.traits.preferred_light) * 0.6
        if material == life.traits.preferred_material:
            score += 2.5
        plants = self.world.plants[y][x]
        if life.traits.diet in (DietType.HERBIVORE, DietType.OMNIVORE):
            score += plants * 4
        if life.traits.diet == DietType.PHOTOTROPH:
            score += light_level * 5
        return score

    def move_lifeform(self, life: Lifeform) -> None:
        best_x, best_y = life.x, life.y
        best_score = self.evaluate_cell_suitability(life, life.x, life.y)
        for _ in range(life.traits.speed):
            candidates: List[Tuple[int, int]] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = life.x + dx, life.y + dy
                    if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                        candidates.append((nx, ny))
            if not candidates:
                break
            random.shuffle(candidates)
            for nx, ny in candidates:
                score = self.evaluate_cell_suitability(life, nx, ny)
                if score > best_score + random.uniform(-0.5, 0.5):
                    best_score = score
                    best_x, best_y = nx, ny
            life.x, life.y = best_x, best_y
            best_score = self.evaluate_cell_suitability(life, life.x, life.y)

        if random.random() < 0.005 * life.traits.metabolism:
            life.alive = False
            life.energy = 0.0

    def handle_predation(self, cell_map: Dict[Tuple[int, int], List[Lifeform]]) -> None:
        for occupants in cell_map.values():
            if len(occupants) < 2:
                continue
            carnivores = [o for o in occupants if o.traits.diet in (DietType.CARNIVORE, DietType.OMNIVORE)]
            prey = [o for o in occupants if o.traits.diet in (DietType.HERBIVORE, DietType.PHOTOTROPH)]
            if not carnivores or not prey:
                continue
            random.shuffle(prey)
            for hunter in carnivores:
                if not prey:
                    break
                target = prey.pop()
                if not target.alive:
                    continue
                target.alive = False
                target.energy = 0.0
                self.record_death(target)
                hunter.energy += 45 * hunter.traits.diet_efficiency
                hunter.energy = min(hunter.energy, 220.0)

    def try_reproduce(self, life: Lifeform) -> None:
        if not life.alive:
            return
        if life.energy < life.traits.reproduction_energy or life.age < 18:
            return
        if len(self.lifeforms) + len(self.pending_births) >= MAX_LIFEFORM_COUNT:
            return
        neighbours = self.available_neighbours(life.x, life.y)
        if not neighbours:
            return
        nx, ny = random.choice(neighbours)
        mutation = life.traits.mutation_rate
        child_genes = mutate_genes(life.genes, mutation)
        child_traits = interpret_genes(child_genes)
        child = Lifeform(
            identifier=self.next_life_identifier,
            x=nx,
            y=ny,
            genes=child_genes,
            traits=child_traits,
            species_name=matrix_to_species_name(child_genes),
            color=genes_to_color(child_genes),
            energy=life.energy * 0.45 + random.uniform(10.0, 25.0),
        )
        self.next_life_identifier += 1
        self.pending_births.append(child)
        life.energy *= 0.45
        life.children += 1

    def available_neighbours(self, x: int, y: int) -> List[Tuple[int, int]]:
        locations: List[Tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                    locations.append((nx, ny))
        return locations

    def update_species_populations(self) -> None:
        counts: Dict[str, int] = defaultdict(int)
        for life in self.lifeforms:
            if life.alive:
                counts[life.species_name] += 1
        for species in self.species_registry.values():
            species.population = counts.get(species.name, 0)
            species.historical_max = max(species.historical_max, species.population)

    # ------------------------------------------------------------------
    # User interaction helpers
    # ------------------------------------------------------------------

    def paint_material(self, x: int, y: int) -> None:
        self.world.set_material(x, y, self.selected_material)

    def scatter_lifeforms(self, count: int = 12) -> None:
        for _ in range(count):
            x = random.randrange(self.world.width)
            y = random.randrange(self.world.height)
            self.create_lifeform(x, y, generate_random_genes())

    def spawn_life_at(self, x: int, y: int, diet: Optional[DietType] = None) -> None:
        self.create_lifeform(x, y, generate_random_genes(diet))

    def reset(self) -> None:
        self.world.reset()
        self.lifeforms.clear()
        self.pending_births.clear()
        self.species_registry.clear()
        self.next_life_identifier = 1


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


class Renderer:
    def __init__(self, simulation: Simulation, screen: pygame.Surface) -> None:
        self.simulation = simulation
        self.screen = screen
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)

    def draw(self, show_encyclopedia: bool) -> None:
        self.draw_world()
        self.draw_lifeforms()
        if show_encyclopedia:
            self.draw_encyclopedia()
        self.draw_panel(show_encyclopedia)

    def draw_world(self) -> None:
        world = self.simulation.world
        for y in range(world.height):
            for x in range(world.width):
                material = world.tiles[y][x]
                props = MATERIAL_PROPERTIES[material]
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, props.color, rect)
                vegetation = world.plants[y][x]
                if vegetation > 0.05:
                    green = clamp(80 + vegetation * 120, 0, 200)
                    overlay = (40, int(green), 40)
                    pygame.draw.rect(self.screen, overlay, rect, 0)
                light = clamp(world.light[y][x], 0.0, 1.2)
                shade = int((1.2 - light) * 60)
                if shade > 0:
                    pygame.draw.rect(self.screen, (shade, shade, shade), rect, 1)

    def draw_lifeforms(self) -> None:
        for life in self.simulation.lifeforms:
            if not life.alive:
                continue
            cx = life.x * CELL_SIZE + CELL_SIZE // 2
            cy = life.y * CELL_SIZE + CELL_SIZE // 2
            radius = max(3, min(CELL_SIZE // 2 - 1, 3 + life.traits.speed))
            pygame.draw.circle(self.screen, life.color, (cx, cy), radius)
            if life.traits.diet == DietType.CARNIVORE:
                pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), radius, 2)

    def draw_panel(self, show_encyclopedia: bool) -> None:
        panel_rect = pygame.Rect(0, GRID_HEIGHT * CELL_SIZE, SCREEN_WIDTH, PANEL_HEIGHT)
        pygame.draw.rect(self.screen, (24, 24, 32), panel_rect)
        pygame.draw.rect(self.screen, (60, 60, 80), panel_rect, 2)

        lines = self.generate_status_lines(show_encyclopedia)
        for idx, text in enumerate(lines):
            surface = self.font.render(text, True, (220, 220, 230))
            self.screen.blit(surface, (12, GRID_HEIGHT * CELL_SIZE + 12 + idx * 18))

    def generate_status_lines(self, show_encyclopedia: bool) -> List[str]:
        world = self.simulation.world
        population = len([life for life in self.simulation.lifeforms if life.alive])
        unique_species = len([s for s in self.simulation.species_registry.values() if s.population > 0])
        material_name = self.simulation.selected_material.name.title()
        lines = [
            f"Tick {world.tick:05d} | Population {population} | Species {unique_species} | Global light {world.global_light:.2f}",
            f"Selected material: {material_name}. Controls: 1-4 paint, P/H/C/O spawn, L scatter, Space pause, E encyclopedia.",
            "Hold the spawn key and left-click to place organisms. Right-click clears vegetation.",
        ]
        if show_encyclopedia:
            lines.append("Encyclopedia open – use arrow keys or mouse wheel to scroll.")
        else:
            lines.append("Press E to open the species encyclopedia.")
        return lines

    def draw_encyclopedia(self) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((12, 12, 18, 220))
        self.screen.blit(overlay, (0, 0))
        title = self.font.render("Species Encyclopedia", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))

        sorted_species = sorted(
            self.simulation.species_registry.values(),
            key=lambda s: (-s.population, -s.historical_max, s.name),
        )

        start_y = 60 - self.simulation.encyclopedia_scroll
        for species in sorted_species:
            block_rect = pygame.Rect(20, start_y, SCREEN_WIDTH - 40, 64)
            color_rect = pygame.Rect(block_rect.x, block_rect.y, 14, block_rect.height)
            pygame.draw.rect(self.screen, species.color, color_rect)
            pygame.draw.rect(self.screen, (80, 80, 100), block_rect, 1)
            text_lines = [
                f"{species.name} | pop {species.population} (max {species.historical_max})",
                f"Diet: {species.traits.diet.name.title()} | Mutation {species.traits.mutation_rate:.3f} | Children born {species.total_births}",
                f"{species.description}",
            ]
            for i, line in enumerate(text_lines):
                surface = self.small_font.render(line, True, (220, 220, 230))
                self.screen.blit(surface, (block_rect.x + 24, block_rect.y + 6 + i * 18))
            matrix_signature_text = matrix_signature(species.genes)
            matrix_surface = self.small_font.render(matrix_signature_text, True, (160, 160, 200))
            self.screen.blit(matrix_surface, (block_rect.x + 24, block_rect.y + 6 + 3 * 18))
            start_y += block_rect.height + 12


# ---------------------------------------------------------------------------
# Application entry point and event loop
# ---------------------------------------------------------------------------


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Aurora Evolutions")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    simulation = Simulation()
    renderer = Renderer(simulation, screen)

    for _ in range(28):
        x = random.randrange(simulation.world.width)
        y = random.randrange(simulation.world.height)
        simulation.create_lifeform(x, y, generate_random_genes())

    show_encyclopedia = False
    spawn_mode: Optional[DietType] = None

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    simulation.paused = not simulation.paused
                elif event.key == pygame.K_e:
                    show_encyclopedia = not show_encyclopedia
                elif event.key == pygame.K_r:
                    simulation.reset()
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    index = event.key - pygame.K_1
                    simulation.selected_material = MATERIAL_ORDER[index]
                elif event.key == pygame.K_l:
                    simulation.scatter_lifeforms()
                elif event.key == pygame.K_p:
                    spawn_mode = DietType.PHOTOTROPH
                elif event.key == pygame.K_h:
                    spawn_mode = DietType.HERBIVORE
                elif event.key == pygame.K_c:
                    spawn_mode = DietType.CARNIVORE
                elif event.key == pygame.K_o:
                    spawn_mode = DietType.OMNIVORE
                elif event.key == pygame.K_TAB:
                    spawn_mode = None
                elif show_encyclopedia and event.key == pygame.K_UP:
                    simulation.encyclopedia_scroll = max(0, simulation.encyclopedia_scroll - 40)
                elif show_encyclopedia and event.key == pygame.K_DOWN:
                    simulation.encyclopedia_scroll += 40
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_p, pygame.K_h, pygame.K_c, pygame.K_o):
                    spawn_mode = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                grid_x = mx // CELL_SIZE
                grid_y = my // CELL_SIZE
                if event.button == 1:
                    if my < GRID_HEIGHT * CELL_SIZE:
                        if spawn_mode is None:
                            simulation.paint_material(grid_x, grid_y)
                        else:
                            simulation.spawn_life_at(grid_x, grid_y, spawn_mode)
                    else:
                        show_encyclopedia = False
                elif event.button == 3 and my < GRID_HEIGHT * CELL_SIZE:
                    simulation.world.plants[grid_y][grid_x] = 0.0
                elif show_encyclopedia and event.button == 4:
                    simulation.encyclopedia_scroll = max(0, simulation.encyclopedia_scroll - 40)
                elif show_encyclopedia and event.button == 5:
                    simulation.encyclopedia_scroll += 40

        simulation.update(dt)
        screen.fill((10, 10, 16))
        renderer.draw(show_encyclopedia)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
