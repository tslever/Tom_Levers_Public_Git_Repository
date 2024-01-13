import Displayer from './Displayer';

type Props = {
  label: String;
  onClick: () => void;
};

function ButtonDisplayer(props: Props): JSX.Element {


  return (
    <Displayer>
      <button onClick = { props.onClick }>
        { props.label }
      </button>
    </Displayer>
  );
}

export default ButtonDisplayer;