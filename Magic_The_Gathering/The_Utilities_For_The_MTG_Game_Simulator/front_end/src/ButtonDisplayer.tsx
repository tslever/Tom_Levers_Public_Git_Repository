import Displayer from "./Displayer";

type Props = {
    label: String
};

function TableDisplayer (props: Props): JSX.Element {
    return (
        <Displayer>
            <button>
                { props.label }
            </button>
        </Displayer>
    );
};

export default TableDisplayer;