"""
Create a standardized glomeruli volume dataset.
Generate an independent CSV file per sample containing the true volume of
every glomerulus.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bombusal.config import data_root, results_dir

class GlomeruliVolumeProcessor:
    def __init__(self, data_root, summary_csv, output_dir):
        self.data_root = Path(data_root)
        self.summary_df = pd.read_csv(summary_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.processing_log = []

    def extract_sample_info(self, seg_counts_path):
        """Extract sample info from the file path."""
        parts = seg_counts_path.parts
        species = parts[-3]
        sample = parts[-2]
        al_type = seg_counts_path.stem.split('_')[0]  # AL1 or AL2
        return species, sample, al_type

    def get_voxel_size(self, species, sample, al_type):
        """Get the voxel size for this sample."""
        match = self.summary_df[
            (self.summary_df['species'] == species) &
            (self.summary_df['sample'] == sample) &
            (self.summary_df['al_type'] == al_type)
        ]

        if match.empty:
            raise ValueError(f"No match found: {species}/{sample}/{al_type}")

        total_voxels = match.iloc[0]['total_voxels']
        real_volume = match.iloc[0]['real_volume_um3']
        voxel_size = real_volume / total_voxels

        return voxel_size, total_voxels, real_volume

    def process_single_sample(self, seg_counts_path):
        """Process a single sample and generate a standardized volume CSV."""
        try:
            # Extract sample info
            species, sample, al_type = self.extract_sample_info(seg_counts_path)
            print(f"\n📊 Processing: {species}/{sample}/{al_type}")

            # Get the voxel size
            voxel_size, total_voxels, real_volume = self.get_voxel_size(
                species, sample, al_type
            )

            # Read segmentation count data
            seg_df = pd.read_csv(seg_counts_path)

            # Filter out background (Label 0)
            seg_df = seg_df[seg_df['Label'] != 0].copy()

            # Build the standardized output DataFrame
            output_df = pd.DataFrame({
                'glomerulus_id': seg_df['Label'].values,
                'voxel_count': seg_df['Count'].values,
                'volume_um3': seg_df['Count'].values * voxel_size,
                'percentage': seg_df['Ratio'].values * 100  # Convert to percentage
            })

            # Sort by volume in descending order
            output_df = output_df.sort_values('volume_um3', ascending=False)
            output_df = output_df.reset_index(drop=True)

            # Build the output filename
            output_filename = f"{species}_{sample}_{al_type}.csv"
            output_path = self.output_dir / output_filename

            # Save CSV
            output_df.to_csv(output_path, index=False)

            # Statistics
            stats = {
                'species': species,
                'sample': sample,
                'al_type': al_type,
                'status': 'success',
                'num_glomeruli': len(output_df),
                'total_volume_um3': output_df['volume_um3'].sum(),
                'mean_volume_um3': output_df['volume_um3'].mean(),
                'median_volume_um3': output_df['volume_um3'].median(),
                'min_volume_um3': output_df['volume_um3'].min(),
                'max_volume_um3': output_df['volume_um3'].max(),
                'voxel_size_um3': voxel_size,
                'output_file': output_filename
            }

            print(f"   ✓ Saved: {output_filename}")
            print(f"   Glomeruli: {stats['num_glomeruli']}")
            print(f"   Total volume: {stats['total_volume_um3']:.2f} μm³")
            print(f"   Mean volume: {stats['mean_volume_um3']:.2f} μm³")

            return stats

        except Exception as e:
            print(f"   ✗ Error: {e}")
            species = 'unknown'
            sample = 'unknown'
            al_type = 'unknown'
            try:
                species, sample, al_type = self.extract_sample_info(seg_counts_path)
            except:
                pass

            return {
                'species': species,
                'sample': sample,
                'al_type': al_type,
                'status': 'failed',
                'error': str(e),
                'file_path': str(seg_counts_path)
            }

    def process_all_samples(self):
        """Process all samples in batch."""
        # Find all seg_counts.csv files
        seg_files = list(self.data_root.rglob("*_seg_counts.csv"))

        print(f"\n🔍 Found {len(seg_files)} seg_counts.csv files\n")
        print("="*70)

        # Process each file
        for i, seg_file in enumerate(seg_files, 1):
            print(f"\n[{i}/{len(seg_files)}]", end=" ")
            stats = self.process_single_sample(seg_file)
            self.processing_log.append(stats)

        # Save the processing log
        self.save_processing_log()

        # Generate summary statistics
        self.generate_summary_statistics()

    def save_processing_log(self):
        """Save the processing log."""
        log_df = pd.DataFrame(self.processing_log)
        log_path = self.output_dir / "processing_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"\n{'='*70}")
        print(f"📝 Processing log saved to: {log_path}")

        # Count successes and failures
        success_count = (log_df['status'] == 'success').sum()
        failed_count = (log_df['status'] == 'failed').sum()
        print(f"✓ Success: {success_count}, ✗ Failed: {failed_count}")

        # If any failed, show details
        if failed_count > 0:
            print(f"\n⚠️  Failed samples:")
            failed_df = log_df[log_df['status'] == 'failed']
            for _, row in failed_df.iterrows():
                print(f"   - {row['species']}/{row['sample']}/{row['al_type']}: {row.get('error', 'Unknown error')}")

    def generate_summary_statistics(self):
        """Generate summary statistics."""
        log_df = pd.DataFrame(self.processing_log)
        success_df = log_df[log_df['status'] == 'success'].copy()

        if success_df.empty:
            print("\n⚠️  No successful samples to summarize")
            return

        # Statistics by species
        species_stats = success_df.groupby('species').agg({
            'num_glomeruli': ['mean', 'std', 'min', 'max'],
            'mean_volume_um3': ['mean', 'std'],
            'total_volume_um3': ['mean', 'std']
        }).round(2)

        stats_path = self.output_dir / "summary_statistics.csv"
        species_stats.to_csv(stats_path)
        print(f"📊 Summary statistics saved to: {stats_path}")

        # Print overview
        print(f"\n{'='*70}")
        print("📈 SUMMARY BY SPECIES")
        print(f"{'='*70}")
        for species in sorted(success_df['species'].unique()):
            species_data = success_df[success_df['species'] == species]
            print(f"\n🐝 {species.upper()}:")
            print(f"   Samples processed: {len(species_data)}")
            print(f"   Mean glomeruli count: {species_data['num_glomeruli'].mean():.1f} ± {species_data['num_glomeruli'].std():.1f}")
            print(f"   Mean AL volume: {species_data['total_volume_um3'].mean():.2f} ± {species_data['total_volume_um3'].std():.2f} μm³")
            print(f"   Mean glomerulus volume: {species_data['mean_volume_um3'].mean():.2f} ± {species_data['mean_volume_um3'].std():.2f} μm³")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("🧠 GLOMERULI VOLUME CALCULATION")
    print("="*70)

    # Configure paths
    DATA_ROOT = str(data_root())
    SUMMARY_CSV = str(results_dir() / "segmentation_analysis_results.csv")
    OUTPUT_DIR = str(results_dir() / "glomeruli_volumes")

    print(f"\n📂 Data root: {DATA_ROOT}")
    print(f"📄 Summary CSV: {SUMMARY_CSV}")
    print(f"💾 Output directory: {OUTPUT_DIR}")

    # Create the processor and run
    processor = GlomeruliVolumeProcessor(DATA_ROOT, SUMMARY_CSV, OUTPUT_DIR)
    processor.process_all_samples()

    print(f"\n{'='*70}")
    print(f"✅ All done! Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
