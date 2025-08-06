from typing import Any, Union

from pyRadPlan.machines import Machine, PhotonLINAC, IonAccelerator


def validate_machine(data: Union[dict[str, Any], Machine, None] = None, **kwargs) -> Machine:
    """
    Create a Machine object.

    Parameters
    ----------
    data : Union[Dict[str, Any], None]
        Dictionary containing the data to create the Machine object.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    Machine
        A Machine object.

    Raises
    ------
    ValueError
        If the machine can not be constructed.
    """

    if data:
        # If data is already a Machine object, return it directly
        if isinstance(data, Machine):
            return data

        # Now we check for a dictionary
        if isinstance(data, dict):
            metadata = data.get("meta")

            if isinstance(metadata, dict):
                # obtain the radiation mode if we have a dictionary at our hands
                radiation_mode = metadata.get("radiation_mode")

                # Since we also allow camelCase, try to get radiationMode
                # if radiation_mode is not set
                if radiation_mode is None:
                    radiation_mode = metadata.get("radiationMode")

                if radiation_mode == "photons":
                    return PhotonLINAC.model_validate(data)

                if radiation_mode in ["protons", "helium", "carbon"]:
                    return IonAccelerator.model_validate(data)

        raise ValueError("Dictionary Structure of provided machine not valid!")

    radiation_mode = kwargs.get("radiation_mode", "")
    if radiation_mode == "photons":
        return PhotonLINAC(**kwargs)

    if radiation_mode in ["protons", "helium", "carbon"]:
        return IonAccelerator(**kwargs)

    raise ValueError(f"Unknown radiation mode: {radiation_mode}")
