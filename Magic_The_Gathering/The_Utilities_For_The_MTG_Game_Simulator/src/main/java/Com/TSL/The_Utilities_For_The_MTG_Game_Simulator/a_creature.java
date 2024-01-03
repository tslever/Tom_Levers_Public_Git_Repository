package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_creature extends a_permanent {
	
	private boolean Indicator_Of_Whether_This_Creature_Has_Been_Controlled_By_The_Active_Player_Continuously_Since_The_Turn_Began = false;
	private boolean Indicator_Of_Whether_This_Creature_Is_Blocked = false;
	private boolean Indicator_Of_Whether_This_Creature_Is_Blocking = false;
	private boolean Indicator_Of_Whether_This_Creature_Will_Attack = false;
	private ArrayList<a_static_ability> List_Of_Static_Abilities;
	private Object Target;
	
	public a_creature(String The_Name, ArrayList<a_static_ability> The_List_Of_Static_Abilities_To_Use) {
		super(The_Name);
		this.List_Of_Static_Abilities = The_List_Of_Static_Abilities_To_Use;
	}
	
	public boolean can_attack() {
		return true;
	}
	
	public boolean has_been_controlled_by_the_active_player_continuously_since_the_turn_began() {
		return this.Indicator_Of_Whether_This_Creature_Has_Been_Controlled_By_The_Active_Player_Continuously_Since_The_Turn_Began;
	}
	
	public boolean has_haste() {
		for (a_static_ability The_Static_Ability : this.List_Of_Static_Abilities) {
			if (The_Static_Ability.effect().equals("haste")) {
				return true;
			}
		}
		return false;
	}
	
	public boolean is_battle() {
		return false;
	}
	
	public boolean is_blocked() {
		return this.Indicator_Of_Whether_This_Creature_Is_Blocked;
	}	
	
	public boolean is_blocking() {
		return this.Indicator_Of_Whether_This_Creature_Is_Blocking;
	}
	
	public ArrayList<a_static_ability> list_of_static_abilities() {
		return this.List_Of_Static_Abilities;
	}
	
	public boolean must_attack() {
		return false;
	}
	
	public void sets_the_creatures_indicator_that_this_creature_has_been_controlled_by_the_active_player_continuously_since_the_turn_began(boolean The_Indicator_To_Use) {
		this.Indicator_Of_Whether_This_Creature_Has_Been_Controlled_By_The_Active_Player_Continuously_Since_The_Turn_Began = The_Indicator_To_Use;
	}
	
	public void sets_the_creatures_indicator_of_whether_this_creature_will_attack(boolean The_Indicator_To_Use) {
		this.Indicator_Of_Whether_This_Creature_Will_Attack = The_Indicator_To_Use;
	}
	
	public void targets(Object The_Target) throws Exception {
		if ((The_Target instanceof a_player) || (The_Target instanceof a_planeswalker)) {
			this.Target = The_Target;
		} else {
			throw new Exception("The target is not a player or a planeswalker.");
		}
	}
}