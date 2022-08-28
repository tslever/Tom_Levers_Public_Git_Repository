import { useCallback } from "react";

type Props = {
    title: string,
    index: number,
    setSelectedTabIndex: (index: number) => void
};

function TabHeaderDisplayer ({ title, index, setSelectedTabIndex }: Props) {
    const onClick: (() => void) = useCallback<() => void>(
        /*callback = */ () => { setSelectedTabIndex(index) },
        /*deps: DependencyList = */ [index, setSelectedTabIndex]
    );
    return (
        <button
            onClick = { onClick }
        >
            { title }
        </button>
    );
};

export default TabHeaderDisplayer;