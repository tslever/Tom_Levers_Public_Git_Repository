package lattice_utilities;


import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;


class WorldTest {

	@Test
	void testDisplay() {
		
		int heightOfWorld = 9;
		
		int widthOfWorld = 16;
		
		World world = new World(heightOfWorld, widthOfWorld);
		
		world.display();
		
	}

}
