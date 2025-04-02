from fuzzy_system import run_fuzzy_control_system


def main():
    # Example inputs
    sludge_input = 60  # Sludge level input
    grease_input = 70  # Grease level input

    # Run the fuzzy control system
    washing_time = run_fuzzy_control_system(sludge_input, grease_input)
    print(f"Predicted washing time: {washing_time} minutes")


if __name__ == "__main__":
    main()
