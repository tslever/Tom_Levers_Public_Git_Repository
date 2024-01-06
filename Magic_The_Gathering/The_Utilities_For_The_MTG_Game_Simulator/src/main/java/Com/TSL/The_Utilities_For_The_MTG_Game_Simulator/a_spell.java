package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


/**
 * a_spell
 * 
 * Rule 405.4: Each spell has all the characteristics of the card associated with it... The controller of a spell is the person who cast it.
 */
public class a_spell {

	private boolean Indicator_Of_Whether_This_Spell_Has_A_Target;
	private String Name;
	private a_nonland_card Nonland_Card;
	private a_player Player;
	private String Type;
	
	public a_spell(String The_Name_To_Use, a_nonland_card The_Nonland_Card, a_player The_Player_To_Use, String The_Type_To_Use) {
		this.Name = The_Name_To_Use;
		this.Nonland_Card = The_Nonland_Card;
		this.Player = The_Player_To_Use;
		this.Type = The_Type_To_Use;
	}
	
	public boolean has_a_target() {
		return this.Indicator_Of_Whether_This_Spell_Has_A_Target;
	}
	
	public String name() {
		return this.Name;
	}
	
	public a_nonland_card nonland_card() {
		return this.Nonland_Card;
	}
	
	public a_player player() {
		return this.Player;
	}
	
	@Override
	public String toString() {
		return this.Name;
	}
	
	public String type() {
		return this.Type;
	}
}
