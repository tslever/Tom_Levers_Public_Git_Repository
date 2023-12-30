package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class a_configuration_of_a_permanent_to_contribute_mana {

	private a_permanent Permanent;
	private int Option_For_Contributing_Mana;
	
	public a_configuration_of_a_permanent_to_contribute_mana(a_permanent The_Permanent_To_Use, int The_Option_For_Contributing_Mana_To_Use) {
		this.Permanent = The_Permanent_To_Use;
		this.Option_For_Contributing_Mana = The_Option_For_Contributing_Mana_To_Use;
	}
	
	public a_permanent provides_its_permanent() {
		return this.Permanent;
	}
	
	public int provides_its_option_for_contributing_mana() {
		return this.Option_For_Contributing_Mana;
	}
	
	@Override
	public String toString() {
		return "{\"Permanent\": " + this.Permanent + ", \"Option_For_Contributing_Mana\": " + this.Option_For_Contributing_Mana + "}";
	}
	
}
