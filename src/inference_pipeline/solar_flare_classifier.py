class SolarFlareClassifier:
    """
    Utility class for converting raw X-ray flux values (W/m^2)
    into the official NOAA flare classification (A, B, C, M, X)
    and providing a human-readable impact description.
    """

    @staticmethod
    def get_flare_class(xray_flux):
        """
        Converts a float flux value (e.g., 5.4e-5) into the
        corresponding NOAA flare label (e.g., 'M5.4').

        Parameters
        ----------
        xray_flux : float
            X-ray flux in W/m^2.

        Returns
        -------
        str
            NOAA flare class string (e.g., 'C3.2', 'X1.0').
        """

        xray_flux = float(xray_flux)


        # NOAA classification based on order of magnitude
        if xray_flux < 1e-7:
            letter = "A"
            multiplier = xray_flux / 1e-8
        elif xray_flux < 1e-6:
            letter = "B"
            multiplier = xray_flux / 1e-7
        elif xray_flux < 1e-5:
            letter = "C"
            multiplier = xray_flux / 1e-6
        elif xray_flux < 1e-4:
            letter = "M"
            multiplier = xray_flux / 1e-5
        else:
            # X-class flares can exceed 9.9 (e.g., the famous X28 event in 2003)
            letter = "X"
            multiplier = xray_flux / 1e-4

        return f"{letter}{multiplier:.1f}"

    @staticmethod
    def get_alert_description(flare_class_string):
        """
        Provides a textual description of the expected Earth impact
        based on the flare class letter.

        Parameters
        ----------
        flare_class_string : str
            NOAA flare class string (e.g., 'C3.2', 'M5.4').

        Returns
        -------
        str
            Human-readable impact description.
        """
        letter = flare_class_string[0].upper()

        descriptions = {
            "A": "Quiet Sun conditions. No impact on Earth.",
            "B": "Background solar activity. Normal space weather.",
            "C": "Minor flare. Possible weak radio disturbances near the poles.",
            "M": "Moderate flare. Possible radio blackouts (R1–R2) on the sunlit side of Earth. "
                 "Potential geomagnetic storms if associated with a CME.",
            "X": "Extreme flare! Strong global radio blackouts (R3–R5). High risk for satellites, "
                 "GPS systems, and terrestrial power grids."
        }

        return descriptions.get(letter, "Unknown flare class.")
