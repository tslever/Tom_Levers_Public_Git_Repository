package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


public class a_spell {

	private String Name;
	private String Type;
	
	public a_spell(String The_Name_To_Use, String The_Type_To_Use) {
		this.Name = The_Name_To_Use;
		this.Type = The_Type_To_Use;
	}
	
	public String provides_its_name() {
		return this.Name;
	}
	
	public String provides_its_type() {
		return this.Type;
	}
	
	@Override
	public String toString() {
		return this.Name;
	}
	
}
