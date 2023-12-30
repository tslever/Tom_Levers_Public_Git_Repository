package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


public abstract class a_permanent {

	private String Name;
	private boolean Tapped_Status;
	private int Number_Of_Options_For_Contributing_Mana = 0;
	
	public a_permanent(String The_Name_To_Use, boolean The_Tapped_Status_To_Use) {
		this.Name = The_Name_To_Use;
		this.Tapped_Status = The_Tapped_Status_To_Use;
	}
	
	public a_mana_pool provides_a_mana_pool_for(int The_Option_For_Contributing_Mana_To_Use) {
		return new a_mana_pool(0, 0, 0, 0, 0, 0);
	}
	
	public a_mana_pool indicates_mana_pool_it_would_contribute_for(int The_Option_For_Contributing_Mana_To_Use) {
		return new a_mana_pool(0, 0, 0, 0, 0, 0);
	}
	
	public boolean indicates_whether_it_is_tapped() {
		return this.Tapped_Status;
	}
	
	public String provides_its_name() {
		return this.Name;
	}
	
	public int provides_its_number_of_options_for_contributing_mana() {
		return this.Number_Of_Options_For_Contributing_Mana;
	}
	
	public void taps() {
		System.out.println(this.Name + " has tapped.");
		this.Tapped_Status = true;
	}
	
	public void untaps() {
		System.out.println(this.Name + " has untapped.");
		this.Tapped_Status = false;
	}
	
	@Override
	public String toString() {
		return this.Name;
	}
}