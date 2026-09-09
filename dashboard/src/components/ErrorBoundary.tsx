import { Component } from "react";
import type { ReactNode } from "react";

type Props = { children: ReactNode };
type State = { message: string | null };

/** Last-resort crash guard: shows a recoverable message instead of a blank page. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { message: null };

  static getDerivedStateFromError(e: unknown): State {
    return { message: e instanceof Error ? e.message : "Unexpected render error" };
  }

  render(): ReactNode {
    if (this.state.message !== null) {
      return (
        <div className="app-shell">
          <div className="alert alert--error" data-testid="app-crash">
            Something went wrong rendering this view ({this.state.message}).{" "}
            <button type="button" onClick={() => this.setState({ message: null })}>
              Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
