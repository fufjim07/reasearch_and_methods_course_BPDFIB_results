import argparse
from synthetic_date import print_synthetic_data_checks
from regression import run_all_analyses
from interaction import run_interaction_analysis
from results_tables import print_results_tables
from visualization import run_prevalence_visualizations
from regression_visualiztaion import run_adjusted_regression_visualizations


def print_section(title):
    """
    Print a clear section title for the console output.
    """

    line = "=" * 80
    print(f"\n{line}")
    print(title)
    print(line)


def main(show_interaction_plot=False, run_tables=True, run_visualizations=True, show_plots=True):
    """
    Run the synthetic data project from data checks to statistical analyses.
    """

    print_section("1. Synthetic data checks")
    print_synthetic_data_checks()

    print_section("2. Statistical tests and main logistic regression")
    run_all_analyses()

    print_section("3. BPD x gender interaction analysis")
    run_interaction_analysis(show_plot=show_interaction_plot)

    if run_tables:
        print_section("4. Results tables")
        print_results_tables(save=True)

    if run_visualizations:
        print_section("5. Visualizations")
        run_prevalence_visualizations(show_plots=show_plots)
        run_adjusted_regression_visualizations(show_plots=show_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the synthetic data generation checks and statistical analyses."
    )
    parser.add_argument(
        "--interaction-plot",
        action="store_true",
        help="Show the BPD x gender interaction plot."
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip the visualization section."
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip the results tables section."
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Print and save only the results tables."
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show visualization windows in addition to saving the PNG files."
    )
    parser.add_argument(
        "--no-show-plots",
        action="store_true",
        help="Save the visualization PNG files without opening graph windows."
    )
    args = parser.parse_args()

    if args.tables_only:
        print_section("Results tables")
        print_results_tables(save=True)
    else:
        main(
            show_interaction_plot=args.interaction_plot,
            run_tables=not args.skip_tables,
            run_visualizations=not args.skip_visualizations,
            show_plots=args.show_plots or not args.no_show_plots
        )
