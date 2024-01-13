import { useState } from 'react';
import './App.css';
import ButtonDisplayer from "./ButtonDisplayer";
import Displayer from "./Displayer";
import TableDisplayer from "./TableDisplayer";

/** Organizer */
function App() {
  
  /**
   * CaretakerOfBodyData is responsible for and controls access to bodyData.
   * A Caretaker is a nexus for information transfer.
   * CaretakerOfBodyData provide notifications of when bodyData changes through calling setBodyData.
  */
  function CaretakerOfBodyData() {
    const [bodyData, setBodyData] = useState<string[][]>([]);
    const onClick = async () => {
      try {
        const response = await fetch('http://localhost:8080');
        const text = await response.text();
        const rowWithText = [text];
        const newBodyData = [];
        newBodyData.push(rowWithText);
        for (let i = 0; i < bodyData.length; i++) {
          newBodyData.push(bodyData[i]);
        }
        setBodyData(newBodyData);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };
    return {
      "bodyData": bodyData,
      "onClick": onClick
    }
  }
  
  /** No concrete objects should exist except those made from a Caretaker by an Organizer. */
  const jsonObjectOfPropertiesOfCaretakerOfBodyData = CaretakerOfBodyData();

  /**
   * Displayers take care of the interface with a user.
   * A Displayer relates to a Caretaker.
   * A displayer formats information obtained from a Caretaker.
   * A displayer updates the display of information when a Caretaker indicates that the Caretaker's information has changed.
   * A displayer handles selections and/or data entry from a user.
   * A displayer provides choices and/or entered data to a Caretaker.
   * For example, a button provides the text of a response when it is clicked; the text is provided to CaretakerOfBodyData.
   * A Displayer is a React component.
   * A Displayer can and should be comprised of more-specific Displayers.
   * Each Displayer is cohesive and loosely coupled.
   * A Displayer does not expose a Caretaker to the Displayer's environment.
   * A Displayer D is assigned the Caretaker with which it interacts at run time by an Organizer (for a top-level Displayer in the Organizer's purview) or from a Displayer that contains D.
   * A Displayer provides a HTML division.
   * A Displayer is constrained by properties.
   * A Displayer D sizes and positions its own Displayers or HTML relative to D's HTML division.
   */
  return (
    <Displayer>
      <ButtonDisplayer
        label = "Next"
        onClick = { jsonObjectOfPropertiesOfCaretakerOfBodyData.onClick }
      />
      <TableDisplayer
        headerData = { ["Action"] }
        bodyData = { jsonObjectOfPropertiesOfCaretakerOfBodyData.bodyData }
      />
    </Displayer>
  );
}

export default App;