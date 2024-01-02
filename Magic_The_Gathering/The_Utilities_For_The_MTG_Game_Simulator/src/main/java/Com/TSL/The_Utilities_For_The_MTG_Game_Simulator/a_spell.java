package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


/**
 * Rule 405.4: Each spell has all the characteristics of the card associated with it... The controller of a spell is the person who cast it.
 *
 */
public class a_spell {

	private String Name;
	private boolean Indicator_Of_Whether_This_Spell_Has_A_Target = false;
	private a_player Player;
	private String Type;
	
	public a_spell(String The_Name_To_Use, a_player The_Player_To_Use, String The_Type_To_Use) {
		this.Player = The_Player_To_Use;
		this.Name = The_Name_To_Use;
		this.Type = The_Type_To_Use;
	}
	
	public boolean has_a_target() {
		return this.Indicator_Of_Whether_This_Spell_Has_A_Target;
	}
	
	public a_player provides_its_player() {
		return this.Player;
	}
	
	public String provides_its_name() {
		return this.Name;
	}
	
	public String type() {
		return this.Type;
	}
	
	@Override
	public String toString() {
		return this.Name;
	}
}
