package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class a_mana_pool {

	private int Number_Of_Black_Mana;
	private int Number_Of_Blue_Mana;
	private int Number_Of_Colorless_Mana;
	private int Number_Of_Green_Mana;
	private int Number_Of_Red_Mana;
	private int Number_Of_White_Mana;
	
	public a_mana_pool(
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
	
	public void decreases_by(a_mana_cost The_Mana_Cost_To_Use) {
		this.Number_Of_Black_Mana -= The_Mana_Cost_To_Use.number_of_black_mana();
		this.Number_Of_Blue_Mana -= The_Mana_Cost_To_Use.number_of_blue_mana();
		this.Number_Of_Colorless_Mana -= The_Mana_Cost_To_Use.number_of_colorless_mana();
		this.Number_Of_Green_Mana -= The_Mana_Cost_To_Use.number_of_green_mana();
		this.Number_Of_Red_Mana -= The_Mana_Cost_To_Use.number_of_red_mana();
		this.Number_Of_White_Mana -= The_Mana_Cost_To_Use.number_of_white_mana();
	}
	
	public void increases_by(a_mana_pool The_Mana_Pool_To_Use) {
		this.Number_Of_Black_Mana += The_Mana_Pool_To_Use.number_of_black_mana();
		this.Number_Of_Blue_Mana += The_Mana_Pool_To_Use.number_of_blue_mana();
		this.Number_Of_Colorless_Mana += The_Mana_Pool_To_Use.number_of_colorless_mana();
		this.Number_Of_Green_Mana += The_Mana_Pool_To_Use.number_of_green_mana();
		this.Number_Of_Red_Mana += The_Mana_Pool_To_Use.number_of_red_mana();
		this.Number_Of_White_Mana += The_Mana_Pool_To_Use.number_of_white_mana();
	}
	
	public boolean is_sufficient_for(a_mana_cost The_Mana_Cost) {
		if (
			this.Number_Of_Black_Mana >= The_Mana_Cost.number_of_black_mana() &&
			this.Number_Of_Blue_Mana >= The_Mana_Cost.number_of_blue_mana() &&
			this.Number_Of_Green_Mana >= The_Mana_Cost.number_of_green_mana() &&
			this.Number_Of_Red_Mana >= The_Mana_Cost.number_of_red_mana() &&
			this.Number_Of_White_Mana >= The_Mana_Cost.number_of_white_mana() &&
			this.number_of_mana() >= The_Mana_Cost.number_of_mana()
		) {
			return true;
		} else {
			return false;
		}
	}

	public int number_of_black_mana() {
		return this.Number_Of_Black_Mana;
	}

	public int number_of_blue_mana() {
		return this.Number_Of_Blue_Mana;
	}

	public int number_of_colorless_mana() {
		return this.Number_Of_Colorless_Mana;
	}
	
	public int number_of_green_mana() {
		return this.Number_Of_Green_Mana;
	}
	
	public int number_of_mana() {
		return
			this.Number_Of_Black_Mana +
			this.Number_Of_Blue_Mana +
			this.Number_Of_Colorless_Mana +
			this.Number_Of_Green_Mana +
			this.Number_Of_Red_Mana +
			this.Number_Of_White_Mana;
	}
	
	public int number_of_red_mana() {
		return this.Number_Of_Red_Mana;
	}
	
	public int number_of_white_mana() {
		return this.Number_Of_White_Mana;
	}
	
	@Override
	public String toString() {
		return
			"{" +
				"\"Number_Of_Black_Mana\": " + this.Number_Of_Black_Mana +
				", \"Number_Of_Blue_Mana\": " + this.Number_Of_Blue_Mana +
				", \"Number_Of_Colorless_Mana\": " + this.Number_Of_Colorless_Mana +
				", \"Number_Of_Green_Mana\": " + this.Number_Of_Green_Mana +
				", \"Number_Of_Red_Mana\": " + this.Number_Of_Red_Mana +
				", \"Number_Of_White_Mana\": " + this.Number_Of_White_Mana +
			"}";
	}
}