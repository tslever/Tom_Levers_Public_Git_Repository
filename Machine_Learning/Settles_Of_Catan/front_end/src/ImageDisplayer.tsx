import Desert from "./Desert.png";
import Displayer from "./Displayer";

type Props = {
    
};

function TableDisplayer (props: Props): JSX.Element {
    return (
        <Displayer>
            <img src = {Desert}/>
        </Displayer>
    );
};

export default TableDisplayer;