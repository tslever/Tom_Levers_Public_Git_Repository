package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class a_mana_cost {

	private int Number_Of_Black_Mana;
	private int Number_Of_Blue_Mana;
	private int Number_Of_Colorless_Mana;
	private int Number_Of_Green_Mana;
	private int Number_Of_Red_Mana;
	private int Number_Of_White_Mana;
	
	public a_mana_cost(
		int The_Number_Of_Black_Mana_To_Use,
		int The_Number_Of_Blue_Mana_To_Use,
		int The_Number_Of_Colorless_Mana_To_Use,
		int The_Number_Of_Green_Mana_To_Use,
		int The_Number_Of_Red_Mana_To_Use,
		int The_Number_Of_White_Mana_To_Use
	)
	{		
		this.Number_Of_Black_Mana = The_Number_Of_Black_Mana_To_Use;
		this.Number_Of_Blue_Mana = The_Number_Of_Blue_Mana_To_Use;
		this.Number_Of_Colorless_Mana = The_Number_Of_Colorless_Mana_To_Use;
		this.Number_Of_Green_Mana = The_Number_Of_Green_Mana_To_Use;
		this.Number_Of_Red_Mana = The_Number_Of_Red_Mana_To_Use;
		this.Number_Of_White_Mana = The_Number_Of_White_Mana_To_Use;
	}
	
	public int provides_its_number_of_black_mana() {
		return this.Number_Of_Black_Mana;
	}

	public int provides_its_number_of_blue_mana() {
		return this.Number_Of_Blue_Mana;
	}

	public int provides_its_number_of_colorless_mana() {
		return this.Number_Of_Colorless_Mana;
	}
	
	public int provides_its_number_of_green_mana() {
		return this.Number_Of_Green_Mana;
	}
	
	public int provides_its_number_of_red_mana() {
		return this.Number_Of_Red_Mana;
	}
	
	public int provides_its_number_of_white_mana() {
		return this.Number_Of_White_Mana;
	}
}