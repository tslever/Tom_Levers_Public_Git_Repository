import { useState } from 'react';
import './App.css';
import ButtonDisplayer from "./ButtonDisplayer";
import Displayer from "./Displayer";
import TableDisplayer from "./TableDisplayer";

/**
 * An Organizer organizes, initializes, and connects as appropriate all of the Caretakers, Displayers, and Executors that comprise an application.
 * An Organizer provides an Executor a Caretaker with which the Executor can interact.
 * An Organizer provides a Presenter a Caretaker with which the Executor can interact.
 * Each Caretaker can be provided to as many Executors and Displayers as needed.
 * There is one Organizer per application.
 * An organizer represents an application.
 * No concrete objects should exist except those made from a Caretaker by an Organizer.
 * An organizer is a React component.
 * An Organizer is a self-contained and autonomous set of behaviors like an old-style executable.
 * An Organizer may contain Displayers.
 * An Organizer does not have properties.
 * An Organizer is bounded by a Shell.
 * An Organizer configures its Displayers with width and position relative to Shell.
 */
function App() {
  
  /*
    * An Executor takes action.
    * An executor can interact with external entities but not a user.
    * An external entity may be an application server or a database server.
    * An Executor interacts with one or more Caretakers.
    * An Executor waits for the Caretakers with which it interacts to provide notice that a piece of information has changed to new information.
    * When an Executor receives notice, the Executor takes appropriate action based on the new information.
    * The Executor obtains information during its action.
    * The Executor provides information to the appropriate Caretakers.
    * An Executor only interacts with Caretakers.
    * Only An Executor may communicate outside of the Organizer's scope (e.g., through communication with networks, physical devices, and databases).
    * No concrete Executors should exist except those created by an Organizer.
    * An Executor works with agency delegated to them by an Organizer.
    * An Executor is not a React component.
  */
  async function Executor() {
    const response = await fetch('http://localhost:8080');
    const text = await response.text();
    const rowWithText = [text];
    const newBodyData = [];
    newBodyData.push(rowWithText);
    for (let i = 0; i < bodyData.length; i++) {
      newBodyData.push(bodyData[i]);
    }
    setBodyData(newBodyData);
  }

  /**
   * A Caretaker holds, takes care, is responsible for, and controls access to discrete pieces of information.
   * A Caretaker manages modification of its information.
   * A Caretaker is a nexus for information transfer.
   * A Caretaker provides notifications when the Caretaker's information changes.
   * A Caretaker provides new values of information.
   * A Caretaker interacts with Organizers, Displayers, Executors, and Shells.
   * A Caretaker is not a React component.
   * CaretakerOfBodyData is responsible for and controls access to bodyData.
   * CaretakerOfBodyData provide notifications of when bodyData changes through calling setBodyData.
   * CaretakerOfBodyData provides new values of bodyData through calling setBodyData.
  */
  const [bodyData, setBodyData] = useState<string[][]>([]);

  function CaretakerOfIndicatorThatButtonHasBeenClicked() {
    const onClick = async () => {
      await Executor();
    };
    return onClick
  }

  /**
   * A Displayers takes care of a user and the interface with the user.
   * A Displayer interacts with one or more Caretakers to obtain information that the Displayer presents to the user.
   * A Displayer provides the its Caretakers with information received from the user through selection, data entry, or other actions.
   * A Displayer formats information obtained from Caretakers with which it interacts. 
   * A Displayer updates the display of information when a Caretaker indicates that the Caretaker's information has changed.
   * A Displayer handles selections and/or data entry from a user.
   * A Displayer provides choices and/or entered data to a Caretaker.
   * A Displayer is a React component.
   * A Displayer can and should be comprised of more-specific Displayers.
   * Each Displayer is cohesive and loosely coupled.
   * A Displayer does not expose a Caretaker to the Displayer's environment.
   * A Displayer D is assigned the Caretakers with which it interacts at run time by an Organizer (for a top-level Displayer in the Organizer's purview) or from a Displayer that contains D.
   * A Displayer provides a HTML division.
   * A Displayer is constrained by properties.
   * A Displayer D sizes and positions its own Displayers or HTML relative to D's HTML division.
   * For example, a button provides the text of a response when it is clicked; the text is provided to CaretakerOfBodyData.
   * A Displayer is self-bounding.
   */
  return (
    <Displayer>
      <ButtonDisplayer
        label = "Next"
        onClick = { CaretakerOfIndicatorThatButtonHasBeenClicked() }
      />
      <TableDisplayer
        headerData = { ["Action"] }
        bodyData = { bodyData }
      />
    </Displayer>
  );
}

export default App;