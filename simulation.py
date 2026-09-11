"""
Forward-time individual-based evolutionary simulation of a diploid population
under increasing heat stress. Generates a parameter-sweep dataset for ML training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def simulate(
    initial_allele_freq,
    temperature_rate,
    selection_strength,
    food_availability,
    migration_rate,
    population_size,
    generations,
    seed,
    baseline_survival=0.75,
    food_effect=0.18,
    heat_harm=0.35,
    mutation_rate=1e-4,
    carrying_capacity_factor=1.5,
    record_trajectory=False,
):
    """
    Run one replicate of the evolutionary simulation.

    Returns dict with final_allele_frequency, final_population_size, extinct flag,
    and optionally per-generation trajectories.
    """
    rng = np.random.default_rng(seed)
    N = int(population_size)
    base_carrying_capacity = int(N * carrying_capacity_factor)

    # Initialise genotypes: sample allele copies per individual (0, 1, or 2)
    p0 = initial_allele_freq
    allele1 = rng.random(N) < p0  # first allele copy
    allele2 = rng.random(N) < p0  # second allele copy
    genotypes = allele1.astype(int) + allele2.astype(int)  # 0, 1, or 2

    trajectory_freq = []
    trajectory_pop = []

    for gen in range(int(generations)):
        current_N = len(genotypes)
        if current_N < 2:
            # Population extinct
            if record_trajectory:
                trajectory_freq.append(np.nan)
                trajectory_pop.append(current_N)
                for _ in range(gen + 1, int(generations)):
                    trajectory_freq.append(np.nan)
                    trajectory_pop.append(0)
            break

        temperature = temperature_rate * gen

        # --- Survival selection ---
        survival_prob = (
            baseline_survival
            + food_effect * food_availability
            - heat_harm * temperature
            + selection_strength * temperature * (genotypes / 2.0)
        )
        survival_prob = np.clip(survival_prob, 0.03, 0.98)
        survivors = rng.random(current_N) < survival_prob
        genotypes = genotypes[survivors]

        current_N = len(genotypes)
        if current_N < 2:
            if record_trajectory:
                trajectory_freq.append(np.nan)
                trajectory_pop.append(current_N)
                for _ in range(gen + 1, int(generations)):
                    trajectory_freq.append(np.nan)
                    trajectory_pop.append(0)
            break

        # --- Random mating + Mendelian inheritance ---
        # Environment-dependent carrying capacity: degrades with heat, improves with food
        effective_K = max(10, int(base_carrying_capacity * (0.5 + 0.5 * food_availability) / (1.0 + 0.8 * temperature)))
        n_offspring = max(current_N, effective_K)
        parent1_idx = rng.integers(0, current_N, size=n_offspring)
        parent2_idx = rng.integers(0, current_N, size=n_offspring)

        # Each parent donates one allele
        def donate_allele(parent_genotype, rng, n):
            """Given genotype (0,1,2), donate one allele."""
            alleles = np.zeros(n, dtype=int)
            # genotype 0 -> always donate 0
            # genotype 2 -> always donate 1
            # genotype 1 -> 50/50
            mask_het = parent_genotype == 1
            mask_hom_alt = parent_genotype == 2
            alleles[mask_hom_alt] = 1
            alleles[mask_het] = (rng.random(mask_het.sum()) < 0.5).astype(int)
            return alleles

        allele_from_p1 = donate_allele(genotypes[parent1_idx], rng, n_offspring)
        allele_from_p2 = donate_allele(genotypes[parent2_idx], rng, n_offspring)
        offspring_genotypes = allele_from_p1 + allele_from_p2

        # --- Mutation (rare) ---
        for allele_arr in [offspring_genotypes]:
            # Flip alleles with small probability
            flip_mask = rng.random(n_offspring) < mutation_rate
            # For simplicity: if mutation hits, flip one allele direction
            allele_arr[flip_mask] = np.clip(
                allele_arr[flip_mask] + rng.choice([-1, 1], size=flip_mask.sum()),
                0, 2
            )

        genotypes = offspring_genotypes

        # --- Migration ---
        n_migrants = int(len(genotypes) * migration_rate)
        if n_migrants > 0:
            migrant_genotypes = rng.binomial(2, 0.5, size=n_migrants)
            replace_idx = rng.choice(len(genotypes), size=n_migrants, replace=False)
            genotypes[replace_idx] = migrant_genotypes

        # --- Carrying capacity ---
        if len(genotypes) > effective_K:
            keep = rng.choice(len(genotypes), size=effective_K, replace=False)
            genotypes = genotypes[keep]

        # --- Record ---
        if record_trajectory:
            freq = genotypes.sum() / (2 * len(genotypes)) if len(genotypes) > 0 else np.nan
            trajectory_freq.append(freq)
            trajectory_pop.append(len(genotypes))

    # Final state
    final_N = len(genotypes)
    if final_N < 2:
        final_freq = np.nan
        extinct = True
    else:
        final_freq = genotypes.sum() / (2 * final_N)
        extinct = False

    result = {
        "final_allele_frequency": final_freq,
        "final_population_size": final_N,
        "extinct": extinct,
    }
    if record_trajectory:
        result["trajectory_freq"] = trajectory_freq
        result["trajectory_pop"] = trajectory_pop

    return result


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n_scenarios=600, n_replicates=3, base_seed=42):
    rng = np.random.default_rng(base_seed)

    rows = []
    for i in range(n_scenarios):
        params = {
            "initial_allele_frequency": rng.uniform(0.03, 0.70),
            "temperature_rate": rng.uniform(0.002, 0.045),
            "selection_strength": rng.uniform(0.03, 0.35),
            "food_availability": rng.uniform(0.35, 1.00),
            "migration_rate": rng.uniform(0.00, 0.20),
            "population_size": int(rng.integers(90, 281)),
            "generations": int(rng.integers(70, 171)),
        }
        params["final_temperature_anomaly"] = params["temperature_rate"] * params["generations"]

        rep_freqs = []
        rep_pops = []
        rep_extinct = []

        for r in range(n_replicates):
            seed = base_seed * 1000 + i * 100 + r
            result = simulate(
                initial_allele_freq=params["initial_allele_frequency"],
                temperature_rate=params["temperature_rate"],
                selection_strength=params["selection_strength"],
                food_availability=params["food_availability"],
                migration_rate=params["migration_rate"],
                population_size=params["population_size"],
                generations=params["generations"],
                seed=seed,
            )
            rep_freqs.append(result["final_allele_frequency"])
            rep_pops.append(result["final_population_size"])
            rep_extinct.append(result["extinct"])

        # Average across replicates (NaN-aware)
        valid_freqs = [f for f in rep_freqs if not np.isnan(f)]
        avg_freq = np.mean(valid_freqs) if valid_freqs else np.nan
        avg_pop = np.mean(rep_pops)
        ext_prob = np.mean(rep_extinct)

        row = {
            "scenario_id": i,
            **params,
            "simulation_replicates": n_replicates,
            "final_allele_frequency": avg_freq,
            "final_population_size": avg_pop,
            "extinction_probability": ext_prob,
        }
        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_scenarios} scenarios")

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Trajectory plots (Fig 1 & 2)
# ---------------------------------------------------------------------------

def plot_trajectories(output_dir):
    """
    Three scenarios: low warming, rapid warming + good food, rapid warming + low food.
    """
    scenarios = {
        "Low warming": dict(
            initial_allele_freq=0.30, temperature_rate=0.008,
            selection_strength=0.25, food_availability=0.85,
            migration_rate=0.02, population_size=200, generations=120, seed=1
        ),
        "Rapid warming, adequate food": dict(
            initial_allele_freq=0.30, temperature_rate=0.040,
            selection_strength=0.25, food_availability=0.85,
            migration_rate=0.02, population_size=200, generations=120, seed=1
        ),
        "Rapid warming, low food": dict(
            initial_allele_freq=0.30, temperature_rate=0.040,
            selection_strength=0.25, food_availability=0.40,
            migration_rate=0.02, population_size=200, generations=120, seed=1
        ),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, params in scenarios.items():
        result = simulate(**params, record_trajectory=True)
        ax.plot(result["trajectory_freq"], label=label, linewidth=1.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Heat-tolerance allele frequency")
    ax.set_title("Allele Frequency Trajectories Under Different Climate Scenarios")
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "example_trajectories.png"), dpi=150)
    plt.close(fig)
    print("  Saved example_trajectories.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, params in scenarios.items():
        result = simulate(**params, record_trajectory=True)
        ax.plot(result["trajectory_pop"], label=label, linewidth=1.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population size")
    ax.set_title("Population Size Trajectories Under Different Climate Scenarios")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "population_trajectories.png"), dpi=150)
    plt.close(fig)
    print("  Saved population_trajectories.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Generating trajectory plots ===")
    plot_trajectories(OUTPUT_DIR)

    print("=== Generating dataset (600 scenarios x 3 replicates) ===")
    df = generate_dataset(n_scenarios=600, n_replicates=3, base_seed=42)

    csv_path = os.path.join(OUTPUT_DIR, "evolution_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}  ({len(df)} rows)")

    # Summary stats
    print(f"\n  Rows with valid final_allele_frequency: {df['final_allele_frequency'].notna().sum()}")
    print(f"  Rows extinct in all replicates: {(df['extinction_probability'] == 1.0).sum()}")
    print(f"  Mean final allele frequency: {df['final_allele_frequency'].mean():.3f}")
    print(f"  Dataset columns: {list(df.columns)}")
