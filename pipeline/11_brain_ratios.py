"""
Calculate two brain-volume ratios:
1. Each glomerulus as a percentage of the whole brain
2. The whole AL as a percentage of the whole brain
"""

import sys
from pathlib import Path
import re

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bombusal.config import data_root, results_dir

class BrainRatioCalculator:
    def __init__(self, glomeruli_dir, final_results_csv, wholebrain_csv, output_dir):
        self.glomeruli_dir = Path(glomeruli_dir)
        self.final_results_df = pd.read_csv(final_results_csv)
        self.wholebrain_df = pd.read_csv(wholebrain_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Build a map from B_id to whole-brain volume
        self.brain_volume_map = dict(zip(
            self.wholebrain_df['B_id'],
            self.wholebrain_df['real_volume_um3']
        ))

        self.glomeruli_processing_log = []

    def extract_sample_id(self, filename):
        """Extract the sample ID from the filename (e.g. B320, DUR021)."""
        match = re.search(r'(B\d+|DUR\d+)', filename)
        return match.group(1) if match else None

    # ========== Group 1: each glomerulus as a percentage of the whole brain ==========

    def process_glomeruli_brain_ratio(self, glomeruli_csv_path):
        """Add whole-brain percentage to the glomeruli of a single sample."""
        try:
            filename = glomeruli_csv_path.name
            sample_id = self.extract_sample_id(filename)

            if not sample_id or sample_id not in self.brain_volume_map:
                raise ValueError(f"No whole brain volume for {sample_id}")

            whole_brain_volume = self.brain_volume_map[sample_id]

            # Read the glomeruli data
            glom_df = pd.read_csv(glomeruli_csv_path)

            # Compute each glomerulus as a percentage of the whole brain
            glom_df['brain_percentage'] = (glom_df['volume_um3'] / whole_brain_volume) * 100

            # Save the updated CSV
            output_path = self.output_dir / "glomeruli_with_brain_ratio" / filename
            output_path.parent.mkdir(exist_ok=True)
            glom_df.to_csv(output_path, index=False)

            # Summary statistics
            stats = {
                'filename': filename,
                'sample_id': sample_id,
                'status': 'success',
                'num_glomeruli': len(glom_df),
                'al_volume_um3': glom_df['volume_um3'].sum(),
                'whole_brain_volume_um3': whole_brain_volume,
                'mean_glom_brain_pct': glom_df['brain_percentage'].mean(),
                'max_glom_brain_pct': glom_df['brain_percentage'].max()
            }

            print(f"  ✓ {filename}: {len(glom_df)} glomeruli, max brain%: {stats['max_glom_brain_pct']:.6f}%")
            return stats

        except Exception as e:
            print(f"  ✗ Error: {glomeruli_csv_path.name}: {e}")
            return {
                'filename': glomeruli_csv_path.name,
                'status': 'failed',
                'error': str(e)
            }

    def process_all_glomeruli(self):
        """Batch-process all glomeruli files."""
        print(f"\n{'='*70}")
        print("📊 PART 1: Adding brain_percentage to each glomerulus")
        print(f"{'='*70}\n")

        csv_files = [
            f for f in self.glomeruli_dir.glob("*.csv")
            if 'processing_log' not in f.name and 'summary' not in f.name
        ]

        print(f"Found {len(csv_files)} sample files\n")

        for i, csv_file in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}]", end=" ")
            stats = self.process_glomeruli_brain_ratio(csv_file)
            self.glomeruli_processing_log.append(stats)

        # Save the log
        log_df = pd.DataFrame(self.glomeruli_processing_log)
        log_path = self.output_dir / "glomeruli_with_brain_ratio" / "processing_log.csv"
        log_df.to_csv(log_path, index=False)

        success_count = (log_df['status'] == 'success').sum()
        print(f"\n✓ Completed: {success_count}/{len(csv_files)} samples")

    # ========== Group 2: whole AL as a percentage of the whole brain ==========

    def calculate_al_brain_ratios(self):
        """Compute AL/whole-brain percentage for all samples."""
        print(f"\n{'='*70}")
        print("📊 PART 2: Calculating AL/whole_brain ratios")
        print(f"{'='*70}\n")

        # Copy the final_results data
        results_df = self.final_results_df.copy()

        # Add the whole-brain volume column
        results_df['whole_brain_volume_um3'] = results_df['sample'].map(self.brain_volume_map)

        # Compute the AL as a percentage of the whole brain
        results_df['al_brain_percentage'] = (
            results_df['algorithm_volume_um3'] / results_df['whole_brain_volume_um3']
        ) * 100

        # Check for missing data
        missing = results_df['whole_brain_volume_um3'].isna().sum()
        if missing > 0:
            print(f"⚠️  Warning: {missing} samples missing whole brain data")
            print("Samples without brain data:")
            print(results_df[results_df['whole_brain_volume_um3'].isna()][['species', 'sample', 'al_type']])

        # Save the full results
        full_output_path = self.output_dir / "al_brain_ratios_full.csv"
        results_df.to_csv(full_output_path, index=False)
        print(f"\n✓ Full results saved to: {full_output_path}")

        # Create a simplified version (only the key columns)
        summary_cols = [
            'species', 'sample', 'al_type', 'gender',
            'algorithm_volume_um3', 'whole_brain_volume_um3', 'al_brain_percentage',
            'algorithm_glomeruli'
        ]
        summary_df = results_df[summary_cols].copy()

        summary_output_path = self.output_dir / "al_brain_ratios_summary.csv"
        summary_df.to_csv(summary_output_path, index=False)
        print(f"✓ Summary saved to: {summary_output_path}")

        # Statistical analysis
        self.analyze_al_brain_ratios(results_df)

        return results_df

    def analyze_al_brain_ratios(self, results_df):
        """Analyze and print AL/whole-brain ratio statistics."""
        # Filter out missing data
        valid_df = results_df.dropna(subset=['al_brain_percentage'])

        print(f"\n{'='*70}")
        print("📈 AL/BRAIN RATIO STATISTICS")
        print(f"{'='*70}\n")

        # Overall statistics
        print(f"Overall statistics (n={len(valid_df)}):")
        print(f"  Mean AL/Brain: {valid_df['al_brain_percentage'].mean():.4f}%")
        print(f"  Std: {valid_df['al_brain_percentage'].std():.4f}%")
        print(f"  Range: {valid_df['al_brain_percentage'].min():.4f}% - {valid_df['al_brain_percentage'].max():.4f}%")

        # Statistics by species
        print(f"\nBy species:")
        species_stats = valid_df.groupby('species')['al_brain_percentage'].agg(['count', 'mean', 'std', 'min', 'max'])

        for species, row in species_stats.iterrows():
            print(f"\n🐝 {species.upper()}:")
            print(f"   Samples: {int(row['count'])}")
            print(f"   Mean: {row['mean']:.4f}% ± {row['std']:.4f}%")
            print(f"   Range: {row['min']:.4f}% - {row['max']:.4f}%")

        # Save the statistics
        stats_path = self.output_dir / "al_brain_ratios_statistics.csv"
        species_stats.to_csv(stats_path)
        print(f"\n✓ Statistics saved to: {stats_path}")


def main():
    """Main entry point."""
    print(f"\n{'='*70}")
    print("🧠 BRAIN RATIO CALCULATOR")
    print(f"{'='*70}")

    # Configure paths
    GLOMERULI_DIR = str(results_dir() / "glomeruli_volumes")
    FINAL_RESULTS_CSV = str(results_dir() / "final_results.csv")
    WHOLEBRAIN_CSV = str(data_root() / "wholebrain_volumes_with_real_volume.csv")
    OUTPUT_DIR = str(results_dir())

    print(f"\n📂 Glomeruli dir: {GLOMERULI_DIR}")
    print(f"📄 Final results: {FINAL_RESULTS_CSV}")
    print(f"📄 Whole brain: {WHOLEBRAIN_CSV}")
    print(f"💾 Output dir: {OUTPUT_DIR}")

    # Create the calculator
    calculator = BrainRatioCalculator(
        GLOMERULI_DIR,
        FINAL_RESULTS_CSV,
        WHOLEBRAIN_CSV,
        OUTPUT_DIR
    )

    # Run both groups of calculations
    # Group 1: whole-brain percentage of each glomerulus
    calculator.process_all_glomeruli()

    # Group 2: whole-brain percentage of the entire AL
    calculator.calculate_al_brain_ratios()

    print(f"\n{'='*70}")
    print(f"✅ ALL DONE!")
    print(f"{'='*70}")
    print(f"\n📁 Output structure:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── glomeruli_with_brain_ratio/  (brain_percentage per glomerulus)")
    print(f"   ├── al_brain_ratios_full.csv     (full AL / whole-brain data)")
    print(f"   ├── al_brain_ratios_summary.csv  (summary)")
    print(f"   └── al_brain_ratios_statistics.csv (statistics)")
    print()


if __name__ == "__main__":
    main()
