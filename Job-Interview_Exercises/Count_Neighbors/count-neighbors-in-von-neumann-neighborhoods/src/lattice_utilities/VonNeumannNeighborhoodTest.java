package lattice_utilities;


import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;


class VonNeumannNeighborhoodTest {

	@Test
	void testDisplay() {
		
		VonNeumannNeighborhood vonNeumannNeighborhood = new VonNeumannNeighborhood(1);
		
		vonNeumannNeighborhood.display();
		
	}

}
