package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public abstract class a_card
{
	private String Expansion;
	private boolean Is_Playable;
	private String Name;
	private String Type;
	
	protected a_card(String The_Expansion_To_Use, String The_Name_To_Use, String The_Type_To_Use) {
		this.Expansion = The_Expansion_To_Use;
		this.Is_Playable = false;
		this.Name = The_Name_To_Use;
		this.Type = The_Type_To_Use;
	}

	public void becomes_not_playable() {
		this.Is_Playable = false;
	}
	
	public void becomes_playable() {
		this.Is_Playable = true;
	}
	
	public boolean is_playable() {
		return this.Is_Playable;
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
